import torch.nn.functional as F
import torch
from torch import nn

import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score, Precision, Recall, MetricCollection
import pandas as pd
from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored
from collections import deque








class FCLayer(nn.Module):
    def __init__(self, in_size=1024, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x
    


class IClassifier(nn.Module):
    def __init__(self, feature_size=1024, output_class=4):
        super(IClassifier, self).__init__()

        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):

        #x = torch.stack(x[0])

        #x = x[0]
        c = self.fc(x.float().view(x.float().shape[0], -1))  # N x C
        return x.float().view(x.float().shape[0], -1), c




class BClassifier(nn.Module):
    def __init__(
        self,
        input_size=1024,
        
        output_class=4,
        dropout_v=0.0,
        nonlinear=True,
        passing_v=False,
    ):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(
                nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh()
            )
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v), nn.Linear(input_size, input_size), nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(
            c, 0, descending=True
        )  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(
            feats, dim=0, index=m_indices[0, :]
        )  # select critical instances, m_feats in shape C x K
        q_max = self.q(
            m_feats
        )  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(
            Q, q_max.transpose(0, 1)
        )  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(
            A
            / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)),
            0,
        )  # normalize attention scores, A in shape N x C,
        B = torch.mm(
            A.transpose(0, 1), V
        )  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B
    


class EHRClassifier(nn.Module):
    def __init__(self, ehr_input_size=2, ehr_output_size=4, temperature=1.0):
        super(EHRClassifier, self).__init__()
        self.embedding = nn.Embedding(2, 4)  # Assuming embedding size of 4, can be adjusted
        self.fc1 = nn.Linear(5, 16)  # 1 continuous + 4 embedding dimensions
        self.fc2 = nn.Linear(16, ehr_output_size)
        self.ln = nn.LayerNorm(ehr_output_size)
        self.sigmoid = nn.Sigmoid()
        self.temperature = temperature
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x_flattened = x.view(x.size(0), -1)
        continuous_var = x_flattened[:, 1].unsqueeze(1).float() 
        categorical_var = x_flattened[:, 0].long()  
        
        embedded_cat = self.embedding(categorical_var)
        x = torch.cat((continuous_var, embedded_cat), dim=1)
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x) / self.temperature
        logits = self.ln(logits)
        probabilities = self.sigmoid(logits)
        return logits, probabilities



class MILNet(nn.Module):
    def __init__(
        self,
        i_class="trans",
        output_class=4,
        ehr_input_size=2,
    ):
        super(MILNet, self).__init__()

        self.i_classifier = IClassifier(output_class=output_class)
        self.b_classifier = BClassifier(output_class=output_class)
        self.ehr_classifier = EHRClassifier(ehr_input_size=ehr_input_size)
        self.ln = nn.LayerNorm(output_class)

    def forward(self, x):
        feats, ehr = x[0], x[1]

        ehr = ehr.view(1, -1) 

        feats_out, classes = self.i_classifier(feats)

        # Bag classifier stream
        prediction_bag, A, B = self.b_classifier(feats_out, classes)

        # EHR classifier stream
        ehr_feats_out, ehr_classes = self.ehr_classifier(ehr)

        return classes, prediction_bag, A, B, ehr_feats_out, ehr_classes
    


class SURVIVMIL(pl.LightningModule):
    def __init__(
        self,
        criterion = nn.CrossEntropyLoss(),
        num_classes=4,
        prob_transform=0.5,
        model_type='SURVIVMIL',
        max_epochs=100,
        log_dir="./CV_logs/SMPeds",
        output_class=1,
        alpha = 0.0, 
        multimodal = None,
        lossweight = None,
        buffer_size = 142,

        **kwargs,

    ):
        super(SURVIVMIL, self).__init__()

        self.save_hyperparameters(ignore=["criterion"])
        # self.lr = 0.00002
        self.lr = 0.0002
        self.criterion = criterion
        self.model = MILNet(output_class=output_class, **kwargs)
        self.calculate_loss = self.calculate_loss_survival
        self.num_classes = num_classes
        self.prob_transform = prob_transform
        self.max_epochs = max_epochs
        self.alpha = alpha 
        self.multimodal = multimodal
        self.lossweight = lossweight
        self.buffer_size = buffer_size


        # buffer streams (train)
        self.predicted_scores_accum = []
        self.predicted_survival_accum = []
        self.event_times_accum = []
        self.event_accum = []

        self.initial_c_index = None

        self.min_samples_for_cindex = 16  # Adjusted to 10

        
        if output_class > 1:
            self.acc = Accuracy(
                task="multiclass", average="macro", num_classes=num_classes
            )
            self.auc = AUROC(
                task="multiclass", num_classes=num_classes, average="macro"
            )
            self.F1 = F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            )
            self.precision_metric = Precision(
                task="multiclass", num_classes=num_classes, average="macro"
            )
            self.recall = Recall(
                task="multiclass", num_classes=num_classes, average="macro"
            )
        else:
            self.acc = Accuracy(task="binary", average="macro")
            self.auc = AUROC(task="binary", num_classes=num_classes, average="macro")
            self.F1 = F1Score(task="binary", num_classes=num_classes, average="macro")
            self.precision_metric = Precision(
                task="binary", num_classes=num_classes, average="macro"
            )
            self.recall = Recall(
                task="binary", num_classes=num_classes, average="macro"
            )

        self.data = [{"count": 0, "correct": 0} for i in range(self.num_classes)]
        self.log_path = log_dir
        metrics = MetricCollection(
            [
                self.acc,
                self.F1,
                self.precision_metric,
                self.recall,
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def on_after_batch_transfer(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        if self.trainer.training:
            x = x
        return x, y
    

    def append_to_buffer(self, buffer, values):
        buffer.extend(values)
        if len(buffer) > self.buffer_size:
            buffer = buffer[-self.buffer_size:]
        return buffer
    

    def configure_optimizers(self):
        # Separate the parameters of the EHR classifier and the B classifier
        ehr_params = list(self.model.ehr_classifier.parameters())
        b_classifier_params = list(self.model.b_classifier.parameters())
        i_classifier_params = list(self.model.i_classifier.parameters())
        # other_params = [p for n, p in self.named_parameters() if p not in ehr_params + b_classifier_params]

        # Define different learning rates for EHR and B classifier
        lr_ehr = 0.002
        lr_b_classifier = 0.0002
        lr_i_classifier = 0.0002


        optimizer = torch.optim.AdamW(
            [
                {"params": ehr_params, "lr": lr_ehr},
                {"params": b_classifier_params, "lr": lr_b_classifier},
                {"params": i_classifier_params, "lr": lr_i_classifier},
            ],
            weight_decay=1e-4,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=min(lr_ehr, lr_b_classifier, lr_i_classifier) / 50
        )
        return [optimizer], [lr_scheduler]   
    

    def calculate_c_index_loss(self):
        if len(self.event_accum) < self.min_samples_for_cindex:
            return self.initial_c_index if self.initial_c_index is not None else 0.0
        
        event_times = np.concatenate(self.event_times_accum)
        events = np.concatenate(self.event_accum)
        pred_maxed = -torch.sum(torch.tensor(self.predicted_survival_accum), dim = 1) # multiclass summation across tensor dimensions 

        c_index = concordance_index_censored((1-events).astype(bool).flatten(), event_times.flatten(), pred_maxed, tied_tol=1e-08)[0]

        loss_c_index = 1.0 - c_index  

        if self.initial_c_index is None:
            self.initial_c_index = loss_c_index

        return loss_c_index
    

    def calculate_loss_survival(self, inputs, labels, num_classes=2):
        feats, ehr = inputs[0], inputs[1]
        label, time_to_event, c = labels[0], labels[1], labels[2]
        output = self.model((torch.squeeze(feats).float(), torch.squeeze(ehr).float()))
        classes, bag_prediction, ehr_pred = output[0], output[1], output[5]
        max_prediction, index = torch.max(classes, 0)

        if self.multimodal == 'all':
            multi_stream_prediction = self.lossweight*bag_prediction + (1 - self.lossweight)*ehr_pred
        elif self.multimodal == 'wsi':
            multi_stream_prediction = bag_prediction
        elif self.multimodal == 'ehr':
            multi_stream_prediction = ehr_pred
        else:
            print('Not a valid modality provided -- either (all, wsi, ehr)')

        if num_classes > 2:
            hazard = torch.sigmoid(multi_stream_prediction)
            y_hat = torch.topk(multi_stream_prediction, 1, dim = 1)[1]
        else:
            hazard = torch.sigmoid(multi_stream_prediction)
            y_hat = hazard > 0.5

        surv_pred = torch.cumprod(1 - hazard, dim = 1)


        loss_bag = self.criterion(bag_prediction, label.view(-1))
        loss_ehr = self.criterion(ehr_pred, label.view(-1))
        loss_max = self.criterion(max_prediction.unsqueeze(0), label.view(-1))

        if self.multimodal == 'all':
            multi_stream_loss = (0.5 * (loss_bag + loss_max)) + (loss_ehr)
        elif self.multimodal == 'wsi':
            multi_stream_loss = loss_bag + loss_max
        elif self.multimodal == 'ehr':
            multi_stream_loss = loss_ehr
        else: 
            print('Not a valid modality provided -- either (all, wsi, ehr)')

        loss = multi_stream_loss

        if len(self.predicted_scores_accum) >= self.min_samples_for_cindex:
            c_index_loss = self.calculate_c_index_loss()
            loss += c_index_loss


        return loss, hazard, bag_prediction, y_hat, surv_pred
    


    def training_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]
        label, time_to_event, event = labels[0], labels[1], labels[2]

        loss, hazard, _, y_hat, surv_pred = self.calculate_loss(inputs, labels)
        
        acc = self.acc(torch.argmax(y_hat.int(), 1), label.view(-1))
       
        self.log("train_loss", 
                 loss, 
                 on_epoch=True, 
                 logger=True)
        
        self.log("train_acc",
                acc,
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,)
    

        self.data[int(label)]["count"] += 1
        self.data[int(label)]["correct"] += y_hat == label

        dic = {
            "loss": loss,
            "acc": acc,
        }


        # accumulating label, time to event, censorship in buffer for concordance loss
        self.predicted_survival_accum = self.append_to_buffer(self.predicted_survival_accum, surv_pred.detach().cpu().numpy())
        self.predicted_scores_accum = self.append_to_buffer(self.predicted_scores_accum, hazard.detach().cpu().numpy())
        self.event_times_accum = self.append_to_buffer(self.event_times_accum, time_to_event.detach().cpu().numpy())
        self.event_accum = self.append_to_buffer(self.event_accum, event.detach().cpu().numpy())

        return dic
    


    def validation_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]
        label, time_to_event, event = labels[0], labels[1], labels[2]

        loss, hazard, bag_prediction, y_hat, surv_pred = self.calculate_loss(inputs, labels)
        acc = self.acc(torch.argmax(y_hat.int(), dim = 1), label.view(-1))

        
        self.log("val_acc", 
                 acc, 
                 on_step=True, 
                 on_epoch=True, 
                 logger=True, 
                 prog_bar=True)
        


        self.data[int(label)]["count"] += 1
        self.data[int(label)]["correct"] += y_hat == label

        results = {
            "logits": bag_prediction,
            "Y_prob": hazard,
            "Y_hat": y_hat,
            "label": label,
            "time_to_event": time_to_event,
            "inputs": inputs,
            "labels": labels,
            "event": event,
            'surv_pred': surv_pred, 
            'val_loss_step':loss,
        }
        self.validation_step_outputs.append(results)

        return results
    

    def on_validation_epoch_end(self):
        logits = torch.cat([x["logits"] for x in self.validation_step_outputs], dim=0)
        events = torch.cat([x["event"] for x in self.validation_step_outputs], dim=0)
        probs = torch.cat([x["Y_prob"] for x in self.validation_step_outputs], dim=0)
        max_probs = torch.stack([x["Y_hat"] for x in self.validation_step_outputs])
        target = torch.stack([x["label"] for x in self.validation_step_outputs], dim=0) # label is the event
        time_to_target = torch.stack([x["time_to_event"] for x in self.validation_step_outputs], dim=0) # this is time to event
        surv_probs = torch.cat([x["surv_pred"] for x in self.validation_step_outputs], dim=0)
        lossvals = torch.stack([x["val_loss_step"] for x in self.validation_step_outputs], dim=0) 

        events_list = (1-np.array(events.cpu())).astype(bool).flatten().tolist()
        time_list_1d = np.array(time_to_target.cpu()).flatten().tolist()
        risklist = np.array(-torch.sum(surv_probs, dim = 1).cpu()).flatten().tolist()

        c_index = concordance_index_censored(events_list, time_list_1d, risklist, tied_tol=1e-08)[0]

        self.log("val_c_index", 
                 c_index, 
                 prog_bar=True, 
                 on_epoch=True,
                 logger=True)
        
        self.log("val_loss", 
                 lossvals.mean(), 
                 prog_bar=True, 
                 on_epoch=True, 
                 logger=True)

        self.log("val_auc", 
                self.auc(probs, target.view(-1)),
                prog_bar=True, 
                on_epoch=True, 
                logger=True)


        self.log_dict(
            self.valid_metrics(torch.argmax(max_probs.int(), dim = 2).squeeze(), target.view(-1)),
            on_epoch=True,
            logger=True,
        )

        self.validation_step_outputs.clear()
        
        # maybe clear on the end of validation? 
        self.predicted_scores_accum.clear()
        self.predicted_survival_accum.clear()
        self.event_times_accum.clear()
        self.event_accum.clear()
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]
        label, time_to_event, event = labels[0], labels[1], labels[2]

        loss, y_prob, bag_prediction, y_hat, surv_pred = self.calculate_loss(inputs, labels)
        acc = self.acc(torch.argmax(y_hat.int(), dim = 1), label.view(-1))


        self.log("test_loss", 
                 loss, 
                 on_step=True, 
                 on_epoch=True, 
                 logger=True,)
        
        self.log(
                "test_acc",
                acc,
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,)
        
        self.data[int(event)]["count"] += 1
        self.data[int(event)]["correct"] += y_hat == event

        results = {
            "logits": bag_prediction,
            "Y_prob": y_prob,
            "Y_hat": y_hat,
            "label": label,
            "time_to_event": time_to_event,
            "inputs": inputs,
            "labels": labels, 
            "event": event,
            'surv_pred': surv_pred, 
        }

        self.test_step_outputs.append(results)

        return results
    

    def on_test_epoch_end(self):
        logits = torch.cat([x["logits"] for x in self.test_step_outputs], dim=0)
        events = torch.cat([x["event"] for x in self.test_step_outputs], dim=0)
        probs = torch.cat([x["Y_prob"] for x in self.test_step_outputs], dim=0)
        max_probs = torch.stack([x["Y_hat"] for x in self.test_step_outputs])
        target = torch.stack([x["label"] for x in self.test_step_outputs], dim=0)
        time_to_target = torch.stack([x["time_to_event"] for x in self.test_step_outputs], dim=0)
        surv_probs = torch.cat([x["surv_pred"] for x in self.test_step_outputs], dim=0)

        events_list = (1-np.array(events.cpu())).astype(bool).flatten().tolist()
        time_list_1d = np.array(time_to_target.cpu()).flatten().tolist()
        
        
        risklist = np.array(-torch.sum(surv_probs, dim = 1).cpu()).flatten().tolist()
        c_index = concordance_index_censored(events_list, time_list_1d, risklist, tied_tol=1e-08)[0]
        print('EVENTS:', events_list, 'TIMES:', time_list_1d, 'RISK_SCORES',risklist)


        total_test_loss = 0.0
        for output in self.test_step_outputs:
            inputs = output['inputs']
            labels = output['labels'] 
            loss, _, _, _, _ = self.calculate_loss(inputs, labels)
            total_test_loss += loss.item()

        avg_test_loss = (total_test_loss/ len(self.test_step_outputs)) 

        self.log(
            "test_loss",
            avg_test_loss, 
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )

        self.log(
            "test_auc",
            self.auc(probs, target.view(-1)), 
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )

        self.log(
            "test_c_index", 
            c_index, 
            prog_bar=True, 
            on_epoch=True, 
            logger=True,
        )


        self.log_dict(
            self.test_metrics(torch.argmax(max_probs.int(), dim =2).squeeze(), target.view(-1)), 
            on_epoch = True, 
            logger = True,
        )


        self.test_step_outputs.clear()
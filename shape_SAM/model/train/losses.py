import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Alpha:
    Fattore di ponderazione utilizzato per bilanciare l'importanza degli esempi positivi (maschere effettive) rispetto a quelli negativi (background dell'immagine).

    Valore variabile tra 0 e 1.

    Il valore indica il peso da dare alla classe positiva nel calcolo della loss.
    La classe negativa avrà un peso pari a 1 - alpha.

    alpha = 0.5:
        Bilancia equamente le classi positive e negative.
        Utile in dataset bilanciati.

    alpha > 0.5:
        Aumenta l'importanza degli esempi della classe positiva (generalmente minoritaria).
        Esempi comuni: alpha tra 0.6 e 0.9.
        Più alpha è vicino a 1, maggiore è l'importanza data alla classe positiva.
        Utile in dataset con un forte squilibrio a favore della classe negativa.

    alpha < 0.5:
        Aumenta l'importanza degli esempi della classe negativa.
        Meno comune, ma può essere utile se la classe negativa è minoritaria e di maggiore interesse.

          
Gamma:
    Fattore di modulazione che aiuta a ridurre il peso degli esempi ben classificati, concentrandosi maggiormente sugli esempi difficili.
    (Particolarmente utile nei casi in cui il modello può facilmente classificare molti esempi corretti, ma ha difficoltà con quelli più complessi)

    Valore intero positivo.

    Il valore indica quanto dare più peso agli esempi difficili rispetto a quelli facili.

    gamma = 0:
        La perdita focale diventa la stessa della cross-entropy loss standard.
        Tutti gli esempi contribuiscono allo stesso modo alla perdita.

    gamma tra 1 e 2:
        Riduce il peso degli esempi ben classificati.
        Valori comuni: gamma = 2 è spesso utilizzato.
        Aiuta a migliorare le prestazioni sugli esempi difficili senza penalizzare eccessivamente quelli facili.

    gamma > 2:
        Aumenta ulteriormente l'importanza degli esempi difficili.
        Utile in dataset con squilibrio molto forte, dove è critico concentrarsi sugli esempi difficili.
        Valori tipici: gamma tra 3 e 5.

fonte: https://arxiv.org/abs/1708.02002
'''


class DiceLoss(nn.Module):
    
    def __init__(self, smooth: int = 1):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, num_masks: int) -> torch.Tensor:
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            num_masks: Number of masks in the batch
        """

        inputs = inputs.squeeze()
        targets = targets.squeeze()

        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)

        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + self.smooth) / (denominator + self.smooth)
        
        return loss.sum() / num_masks



class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.75, gamma: int = 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, num_masks: int) -> torch.Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            num_masks: Number of masks in the batch
            
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            Returns:
                Loss tensor
        """
        inputs = inputs.squeeze()
        targets = targets.squeeze()

        prob = inputs.sigmoid()
        inputs = inputs.flatten(1)
        prob = prob.flatten(1)
        targets = targets.flatten(1)

        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_masks
    

class IoULoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the Intersection over Union (IoU) loss.
        Args:
            pred_mask: A float tensor of arbitrary shape.
                    The predictions for each example.
            gt_mask: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class). CONTROLLARE QUESTI COMMENTI
        """
         
        pred_mask = (pred_mask >= 0.5).float()
        pred_mask = pred_mask.squeeze()
        gt_mask = gt_mask.squeeze()
        
        pred_mask = pred_mask.flatten(1)
        gt_mask = gt_mask.flatten(1)
            
        intersection = (pred_mask * gt_mask).sum(1)
        union = pred_mask.sum(1) + gt_mask.sum(1) - intersection
        epsilon = 1e-7
        batch_iou = intersection / (union + epsilon)

        return batch_iou
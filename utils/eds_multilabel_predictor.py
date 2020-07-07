from typing import Optional,Dict,List

import torch
import logging

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward, Seq2VecEncoder

from allennlp.nn import InitializerApplicator, RegularizerApplicator,util
from allennlp.training.metrics.metric import Metric

from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, MultiLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from overrides import overrides
from allennlp.models.archival import archive_model, load_archive



@Metric.register("multilabel-f1")
class MultiLabelF1Measure(Metric):
    """
    Computes multilabel F1. Assumes that predictions are 0 or 1.
    """
    def __init__(self) -> None:
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

    def __call__(self,
                 predictions: torch.LongTensor,
                 gold_labels: torch.LongTensor):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of 0 and 1 predictions of shape (batch_size, ..., num_labels).
        gold_labels : ``torch.Tensor``, required.
            A tensor of 0 and 1 predictions of shape (batch_size, ..., num_labels).
        """
        self._true_positives += (predictions * gold_labels).sum().item()
        self._false_positives += (predictions * (1 - gold_labels)).sum().item()
        self._true_negatives += ((1 - predictions) * (1 - gold_labels)).sum().item()
        self._false_negatives += ((1 - predictions) * gold_labels).sum().item()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        predicted_positives = self._true_positives + self._false_positives
        actual_positives = self._true_positives + self._false_negatives

        precision = self._true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = self._true_positives / actual_positives if actual_positives > 0 else 0

        if precision + recall > 0:
            f1_measure = 2 * precision * recall / (precision + recall)
        else:
            f1_measure = 0

        if reset:
            self.reset()
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



@DatasetReader.register("nodeproperties")
class NodeProperties(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,\
                 concept_label_indexers: Dict[str, TokenIndexer] = None,
                 ) -> None:
        super().__init__(lazy=False)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._concept_label_indexers = concept_label_indexers or {
            'concept_label': SingleIdTokenIndexer(namespace='concept_label')}



    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'r', encoding='utf8') as tagger_file:
            for line  in tagger_file.readlines():
                line = line.strip('\n')
                items = line.split('\t')
                tokens = items[0]
                tokens = tokens.split()
                labels = items[1]
                p_vs = items[2:]
                # p_v_pair = []
                # for id,i in enumerate(p_vs):
                #     p_v_pair.append([i,id])

                yield self.text_to_instance(tokens,labels,p_vs)


    @overrides
    def text_to_instance(self,tokens,concept_label,tags=None):
        fields: Dict[str, Field] = {}
        token_field = TextField([Token(a) for a in tokens], self._token_indexers)
        fields["tokens"] = token_field
        cp = [Token(concept_label)]
        fields["concept_label"] = TextField(cp, self._concept_label_indexers)

        if tags:
            fields["p_v_labels"] = MultiLabelField(tags)

        return Instance(fields)


@Model.register("multilabel_classification")
class Mulitilabel_classification(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 token_encoder: Seq2VecEncoder,
                 text_field_embedder: TextFieldEmbedder,
                 concept_label_embedder:TextFieldEmbedder,
                 concept_label_encoder: Seq2VecEncoder,
                 classifier_feedforward : FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Mulitilabel_classification, self).__init__(vocab, regularizer)


        self.token_encoder = token_encoder

        self.text_field_embedder = text_field_embedder
        self.concept_label_embedder = concept_label_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")

        self.concept_label_encoder = concept_label_encoder
        self.classifier_feedforward = classifier_feedforward
        self.f1 = MultiLabelF1Measure()
        self.loss = torch.nn.MultiLabelSoftMarginLoss()
        initializer(self)

    def forward(self, tokens, concept_label, p_v_labels=None) -> Dict[str, torch.Tensor]:
        embedded_tokens = self.text_field_embedder(tokens)
        token_mask = util.get_text_field_mask(tokens)
        encoded_tokens = self.token_encoder(embedded_tokens, token_mask)

        embedded_concept_label = self.concept_label_embedder(concept_label)
        concept_label_mask = util.get_text_field_mask(concept_label)
        encoded_concept_label = self.concept_label_encoder(embedded_concept_label,concept_label_mask)

        logits = self.classifier_feedforward(torch.cat([encoded_tokens,encoded_concept_label], dim=-1))
        probs = torch.sigmoid(logits)
        probs = (probs.data > 0.5).long()

        output_dict = {"probabilities": probs}
        if p_v_labels is not None:
            m = p_v_labels.squeeze(-1)
            loss = self.loss(logits, p_v_labels.squeeze(-1).float())
            output_dict["loss"] = loss
            predictions = (logits.data > 0.0).long()
            label_data = p_v_labels.squeeze(-1).data.long()
            self.f1(predictions, label_data)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1 = self.f1.get_metric(reset)
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }



@Predictor.register('multi_label_predictor')
class multiLabelPredictor(Predictor):
    def predict(self,token:str, concept_label: str) -> JsonDict:
        # This method is implemented in the base class.
        return self.predict_json({"concept_label": concept_label,"token":token})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        concept_label= json_dict["concept_label"]
        token = json_dict["token"]
        return self._dataset_reader.text_to_instance(token,concept_label)

# def make_predictions(model: Model, dataset_reader: DatasetReader) \
#         -> List[Dict[str, float]]:
#     """Make predictions using the given model and dataset reader."""
#     predictions = []
#     predictor = multiLabelPredictor(model, dataset_reader)
#     output = predictor.predict('A good movie!')
#     predictions.append({vocab.get_token_from_index(label_id, 'labels'): prob
#                         for label_id, prob in enumerate(output['probs'])})
#     output = predictor.predict('This was a monstrous waste of time.')
#     predictions.append({vocab.get_token_from_index(label_id, 'labels'): prob
#                         for label_id, prob in enumerate(output['probs'])})
#     return predictions

def load_model(model_path):
    archive = load_archive(model_path)
    predictor = multiLabelPredictor(archive.model, NodeProperties())
    vocab_dict = archive.model.vocab.get_index_to_token_vocabulary('labels')
    return (predictor,vocab_dict)



#
#
# if __name__ == '__main__':
#     archive = load_archive('glove/model.tar.gz')
#     predictor = multiLabelPredictor(archive.model,NodeProperties())
#     c = predictor.predict(token='$',concept_label='dollar_n_1')
#     print(archive.model.vocab.get_index_to_token_vocabulary('labels'))


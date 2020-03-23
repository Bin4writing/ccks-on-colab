import collections
import logging
import os
from queue import Queue
from threading import Thread

import pandas as pd
import tensorflow as tf
import h5py
from tensorflow_core.python.keras.backend import get_session
from tensorflow_core.python.training.session_run_hook import SessionRunArgs

import args
import bert
import modeling
import tokenization
from bert.tokenization import bert_tokenization
from optimizers.adamw import AdamW
from optimizers.utils import get_weight_decays
import numpy as np
import tensorflow.keras.backend as K
import click

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0],True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class SimProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'train.csv')
        train_df = pd.read_csv(file_path, encoding='utf-8')
        train_data = []
        for index, train in enumerate(train_df.values):
            guid = 'train-%d' % index
            text_a = tokenization.convert_to_unicode(str(train[0]))
            text_b = tokenization.convert_to_unicode(str(train[1]))
            label = str(train[2])
            train_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return train_data

    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'dev.csv')
        dev_df = pd.read_csv(file_path, encoding='utf-8')
        dev_data = []
        for index, dev in enumerate(dev_df.values):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(dev[0]))
            text_b = tokenization.convert_to_unicode(str(dev[1]))
            label = str(dev[2])
            dev_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return dev_data

    def get_test_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'test.csv')
        test_df = pd.read_csv(file_path, encoding='utf-8')
        test_data = []
        for index, test in enumerate(test_df.values):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(test[0]))
            text_b = tokenization.convert_to_unicode(str(test[1]))
            label = str(test[2])
            test_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return test_data

    def get_sentence_examples(self, questions):
        for index, data in enumerate(questions):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(data[0]))
            text_b = tokenization.convert_to_unicode(str(data[1]))
            label = str(0)
            yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

    def get_labels(self):
        return ['0', '1']


from os import path as op


def create_tokenizer(model_dir=args.model_dir, do_lower_case=args.do_lower_case, name='bert'):
    if name == 'bert':
        bert.bert_tokenization.validate_case_matches_checkpoint(args.do_lower_case,
                                                                op.join(model_dir, 'bert_model.ckpt'))
        return bert_tokenization.FullTokenizer(vocab_file=op.join(model_dir, 'vocab.txt'),
                                               do_lower_case=do_lower_case)
    raise NotImplemented("* available tokenizers: [ bert, ]")


def flatten_layers(root_layer):
    if isinstance(root_layer, tf.keras.layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        for sub_layer in flatten_layers(layer):
            yield sub_layer


def freeze_bert_layers(l_bert):
    """
    Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
    """
    for layer in flatten_layers(l_bert):
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        elif len(layer._layers) == 0:
            layer.trainable = False
        layer.trainable = False
    l_bert.embeddings_layer.trainable = False


ALLOWED_OPTIMIZERS = ['AdamW']


def create_optimizer(init_lr, steps,
                     weight_decays=None,
                     warmup_steps=None,
                     name='AdamW'):
    if name == 'AdamW':
        weight_decay = 0
        # weight_decay_rate = 0.01,
        # beta_1 = 0.9,
        # beta_2 = 0.999,
        # epsilon = 1e-6,
        # exclude_from_weight_decay = )
        if weight_decays is None:
            raise ValueError("* for 'AdamW', param 'weight_decays' is expected.")

        lr_schedule = tf.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=init_lr,
            decay_steps=steps,
            end_learning_rate=.0,
            power=1,
            cycle=False
        )
        return AdamW(learning_rate=lr_schedule,
                     beta_1=0.9, beta_2=0.999,
                     epsilon=1e-6, amsgrad=False,
                     use_cosine_annealing=warmup_steps is not None,
                     total_iterations=steps if warmup_steps is None else warmup_steps,
                     weight_decays=weight_decays)

    raise NotImplemented("* unknown '{}', available optimizers: {}".format(name, ALLOWED_OPTIMIZERS))


EXCLUDE_FROM_WEIGHT_DECAY = ["LayerNorm", "layer_norm", "bias"]
import re


def use_weight_decay(name):
    for k in EXCLUDE_FROM_WEIGHT_DECAY:
        if re.search(k, name):
            return False
    return True


class InitOptVar(tf.estimator.SessionRunHook):
    def __init__(self, var_list, **kwargs):
        super(InitOptVar, self).__init__(**kwargs)
        self._var_list = var_list

    def before_run(self, run_context):
        init_op = tf.compat.v1.variables_initializer(self._var_list)
        print('=' * 5 + 'init_op')
        return SessionRunArgs(init_op)

    def after_create_session(self, session, coord):
        print('=' * 5 + 'session created')


def create_estimator(steps=None, warmup_steps=None, model_dir=args.model_dir, num_labels=args.num_labels,
                     max_seq_len=args.max_seq_len, learning_rate=args.learning_rate, name='bert'):
    def my_auc(labels, predictions):
        auc_metric = tf.keras.metrics.AUC(name="my_auc")
        auc_metric.update_state(y_true=labels, y_pred=tf.argmax(predictions, 1))
        return {'auc': auc_metric}

    if name == 'bert':
        if warmup_steps is None:
            custom_objects = {
                'BertModelLayer': bert.BertModelLayer,
                'AdamW': AdamW
            }
            model = tf.keras.models.load_model(h5py.File(args.keras_model_path), custom_objects=custom_objects)
            estimator = tf.keras.estimator.model_to_estimator(model, model_dir=args.output_dir)
            return estimator, model
        input_token_ids = tf.keras.Input((max_seq_len,), dtype=tf.int32, name='input_ids')
        input_segment_ids = tf.keras.Input((max_seq_len,), dtype=tf.int32, name='segment_ids')
        input_mask = tf.keras.Input((max_seq_len,), dtype=tf.int32, name='input_mask')
        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        l_bert = bert.BertModelLayer.from_params(bert_params)
        bert_output = l_bert(inputs=[input_token_ids, input_segment_ids], mask=input_mask)
        first_token = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
        pooled_output = tf.keras.layers.Dense(units=first_token.shape[-1], activation=tf.math.tanh)(first_token)
        dropout = tf.keras.layers.Dropout(rate=0.1)(pooled_output)
        logits = tf.keras.layers.Dense(units=num_labels, name='label_ids')(dropout)
        output_prob = tf.keras.layers.Softmax(name='output_prob')(logits)
        model = tf.keras.Model(inputs=[input_token_ids, input_segment_ids, input_mask], outputs=[logits])
        model.build(input_shape=[(None, max_seq_len,), (None, max_seq_len,), (None, max_seq_len,)])
        freeze_bert_layers(l_bert)
        bert.load_stock_weights(l_bert, op.join(model_dir, 'bert_model.ckpt'))
        weight_decays = get_weight_decays(model)
        for k, v in weight_decays.items():
            if use_weight_decay(k):
                weight_decays[k] = 0.01
            else:
                del weight_decays[k]
        opt = create_optimizer(
            init_lr=learning_rate,
            steps=steps,
            weight_decays=weight_decays,
            warmup_steps=warmup_steps,
        )
        model.compile(
            optimizer=opt,
            loss={'label_ids': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)},
            # for numerical stability
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        model.summary()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
        config.log_device_placement = False
        exclude_optimizer_variables = r'^((?!(iter_updates|eta_t)).)*$'
        ws = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=op.join(args.output_dir, 'keras'),
            vars_to_warm_start=exclude_optimizer_variables
        )
        estimator = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                          config=tf.estimator.RunConfig(
                                                              model_dir=args.output_dir,
                                                              session_config=config,
                                                          ))
        estimator._warm_start_settings = ws
        return estimator, model
    raise NotImplemented("* available models: [ bert, ]")


class BertSim:

    def __init__(self):
        tf.get_logger().setLevel(logging.INFO)
        self._model_ckpt = None
        self._mode = None
        self._tokenizer = create_tokenizer()
        self._estimator = None
        self._processor = SimProcessor()
        self._input_queue = None
        self._output_queue = None
        self._predict_thread = None

    def _get_tokenizer(self):
        if self._mode is None:
            raise Exception("* mode is not assigned.")

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, val: tf.estimator.ModeKeys):
        self._mode = val
        if self._mode == tf.estimator.ModeKeys.TRAIN:
            self._model_ckpt = args.ckpt_name
        else:
            self._model_ckpt = args.output_dir
        if self._mode == tf.estimator.ModeKeys.PREDICT:
            self._estimator, _ = create_estimator()
            self._input_queue = Queue(maxsize=1)
            self._output_queue = Queue(maxsize=1)
            self._predict_thread = Thread(target=self.predict_from_queue, daemon=True)
            self._predict_thread.start()

    def predict_from_queue(self):
        for i in self._estimator.predict(input_fn=self.queue_predict_input_fn, yield_single_examples=False):
            self._output_queue.put(i)

    def queue_predict_input_fn(self):
        return (tf.data.Dataset.from_generator(
            self.generate_from_queue,
            output_types={
                'input_ids': tf.int32,
                'input_mask': tf.int32,
                'segment_ids': tf.int32,
            },
            output_shapes={
                'input_ids': (None, args.max_seq_len),
                'input_mask': (None, args.max_seq_len),
                'segment_ids': (None, args.max_seq_len)})
                .prefetch(10))

    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""

        for (ex_index, example) in enumerate(examples):
            label_map = {}
            for (i, label) in enumerate(label_list):
                label_map[label] = i

            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[0:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0     0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = label_map[example.label]
            if ex_index < 5:
                #                tf.logging.info("*** Example ***")
                #                tf.logging.info("guid: %s" % (example.guid))
                #                tf.logging.info("tokens: %s" % " ".join(
                #                    [tokenization.printable_text(x) for x in tokens]))
                #                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                #                tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                #                tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                #                tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
                pass

            feature = InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id)

            yield feature

    def generate_from_queue(self):
        while True:
            predict_examples = self._processor.get_sentence_examples(self._input_queue.get())
            features = list(self.convert_examples_to_features(predict_examples, self._processor.get_labels(),
                                                              args.max_seq_len, self._tokenizer))
            yield {
                'input_ids': [f.input_ids for f in features],
                'input_mask': [f.input_mask for f in features],
                'segment_ids': [f.segment_ids for f in features],
            }

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_single_example(self, ex_index, example, label_list, max_seq_length, tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            #            tf.logging.info("*** Example ***")
            #            tf.logging.info("guid: %s" % (example.guid))
            #            tf.logging.info("tokens: %s" % " ".join(
            #                [tokenization.printable_text(x) for x in tokens]))
            #            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #            tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
            pass

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id)
        return feature

    def file_based_convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer, output_file):
        """Convert a set of `InputExample`s to a TFRecord file."""

        writer = tf.io.TFRecordWriter(output_file)

        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

            feature = self.convert_single_example(ex_index, example, label_list,
                                                  max_seq_length, tokenizer)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature([feature.label_id])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

    def file_based_input_fn_builder(self, input_file, seq_length, is_training, batch_size, drop_remainder):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""

        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        }

        name_to_labels = {
            "label_ids": tf.io.FixedLenFeature([], tf.int64),
        }

        def _decode_record(record, name_to_columns):
            """Decodes a record to a TensorFlow example."""
            example = tf.io.parse_example(serialized=record, features=name_to_columns)
            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.cast(t, dtype=tf.int32)
                example[name] = t
            return example

        def input_fn(params):
            """The actual input function."""

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(input_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)
            d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)

            features = d.map(lambda record: _decode_record(record, name_to_features))
            labels = d.map(lambda record: _decode_record(record, name_to_labels))
            return tf.data.Dataset.zip((features, labels))

        return input_fn

    def train(self):
        bert_config = modeling.BertConfig.from_json_file(args.config_name)

        if args.max_seq_len > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (args.max_seq_len, bert_config.max_position_embeddings))

        tf.io.gfile.makedirs(args.output_dir)

        label_list = self._processor.get_labels()

        train_examples = self._processor.get_train_examples(args.data_dir)
        num_train_steps = int(len(train_examples) / args.batch_size * args.num_train_epochs)

        estimator, model = create_estimator(
            steps=num_train_steps,
            warmup_steps=num_train_steps * 0.1
        )

        train_file = os.path.join(args.output_dir, "train.tf_record")
        self.file_based_convert_examples_to_features(train_examples, label_list, args.max_seq_len, self._tokenizer,
                                                     train_file)
        #        tf.logging.info("***** Running training *****")
        #        tf.logging.info("  Num examples = %d", len(train_examples))
        #        tf.logging.info("  Batch size = %d", args.batch_size)
        #        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = self.file_based_input_fn_builder(input_file=train_file, seq_length=args.max_seq_len,
                                                          is_training=True,
                                                          batch_size=args.batch_size,
                                                          drop_remainder=True)

        # early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
        #     estimator,
        #     metric_name='loss',
        #     max_steps_without_decrease=10,
        #     min_steps=num_train_steps)

        # estimator.train(input_fn=train_input_fn, hooks=[early_stopping])
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        feature_columns = [tf.feature_column.numeric_column(x) for x in ['input_ids', 'input_mask', 'segment_ids']]
        serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            tf.feature_column.make_parse_example_spec(feature_columns))
        estimator.export_saved_model(
            export_dir_base=args.output_dir,
            serving_input_receiver_fn=serving_input_fn,
            experimental_mode=tf.estimator.ModeKeys.EVAL)
        model.reset_metrics()
        model.save(args.keras_model_path)

    def eval(self):
        if self._mode is None:
            raise ValueError("Please set the 'mode' parameter")
        eval_examples = self._processor.get_dev_examples(args.data_dir)
        eval_file = os.path.join(args.output_dir, "eval.tf_record")
        label_list = self._processor.get_labels()
        self.file_based_convert_examples_to_features(
            eval_examples, label_list, args.max_seq_len, self._tokenizer, eval_file)

        # tf.logging.info("***** Running evaluation *****")
        # tf.logging.info("  Num examples = %d", len(eval_examples))
        # tf.logging.info("  Batch size = %d", self.batch_size)
        num_eval_steps = len(eval_examples) / args.batch_size
        eval_input_fn = self.file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=args.max_seq_len,
            is_training=False,
            batch_size=args.batch_size,
            drop_remainder=False)

        estimator, model = create_estimator(num_eval_steps)
        var_list = model.optimizer.variables()
        result = estimator.evaluate(input_fn=eval_input_fn, steps=None, hooks=[])

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
            # tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                # tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        feature_columns = [tf.feature_column.numeric_column(x) for x in ['input_ids', 'input_mask', 'segment_ids']]
        serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            tf.feature_column.make_parse_example_spec(feature_columns))
        estimator.export_saved_model(
            export_dir_base=args.output_dir,
            serving_input_receiver_fn=serving_input_fn,
            experimental_mode=tf.estimator.ModeKeys.PREDICT)
        model.reset_metrics()
        model.save(args.keras_model_path)

    def predict(self, sentence1, sentence2):
        if self._mode is None:
            raise ValueError("Please set the 'mode' parameter")
        self._input_queue.put([(sentence1, sentence2)])
        prediction = self._output_queue.get()
        prob = prediction['label_ids']
        return tf.nn.softmax(prob, axis=-1).numpy()[0][1]


@click.command()
@click.option('--mode', default='train')
def main(mode):
    if tf.executing_eagerly():
        print("Eager Execution Enabled")
    sim = BertSim()
    if mode == 'train':
        sim.mode = tf.estimator.ModeKeys.TRAIN
        sim.train()
        mode = 'eval'
    if mode == 'eval':
        sim.mode = tf.estimator.ModeKeys.EVAL
        sim.eval()


if __name__ == '__main__':
    main()

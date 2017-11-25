import tensorflow as tf
import numpy as np
import logging


def _bytes_feature(value):
  if isinstance(value, np.ndarray):
    value = value.tostring()
  elif isinstance(value, (list, tuple, set)):
    value = ','.join([str(v) for v in value])
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def build_tf_example(kvrecord, ktype, allow_missing=False):
  for key in kvrecord:
    assert key in ktype, "key {} not in ktype, cannot determine serialization type"
  stype = {
      'bytes': _bytes_feature,
      'int64': _int64_feature,
      'float': _float_feature
  }
  features = {}
  for key, value in kvrecord.items():
    try:
      features[key] = stype[ktype[key]](value)
    except KeyError as e:
      logging.warning(
          'Failed to save feature {}: Encountered KeyError: {}'.format(key, e))
    except Exception as e:
      logging.warning('Failed to save feature {}: {}'.format(key, e))
    finally:
      if key not in features and not allow_missing:
        logging.error(
            'Failed to save feature {} and not allow missing'.format(key))
        return None
  if not features:
    return None
  return tf.train.Example(features=tf.train.Features(feature=features))


def write_tfrecords(kvrecord_list, ktype, output_path, allow_missing=False):
  writer = tf.python_io.TFRecordWriter(output_path)
  count = 0
  for kvrecord in kvrecord_list:
    example = build_tf_example(kvrecord, ktype, allow_missing=allow_missing)
    if example:
      writer.write(example.SerializeToString())
      count += 1
  writer.close()
  logging.info('Saved {} tfrecords to {}'.format(count, output_path))


def save_tfrecord_file(kvrecord_list,
                       output_prefix,
                       idx=None,
                       allow_missing=False):
  if idx is not None:
    output_path = '%s_%d.tfrecords' % (output_prefix, idx)
  else:
    output_path = '%s.tfrecords' % output_prefix
  # determine ktype
  ktype = {}
  for key, value in kvrecord_list[0].items():
    if isinstance(value, int):
      ktype[key] = 'int64'
    elif isinstance(value, float):
      ktype[key] = 'float'
    else:
      ktype[key] = 'bytes'
  return write_tfrecords(
      kvrecord_list, ktype, output_path, allow_missing=False)


def read_tfrecords(path, ktype, allow_missing=False):
  reader = tf.python_io.tf_record_iterator(path=path)
  kvrecord_list = []
  for string_record in reader:
    example = tf.train.Example()
    example.ParseFromString(string_record)

    kvrecord = {}
    for key, tf_type in ktype.items():
      try:
        feature = getattr(example.features.feature[key], '%s_list' % tf_type,
                          None).value[0]
        if tf_type == 'bytes':
          feature = np.fromstring(feature)
        kvrecord[key] = feature
      except KeyError as e:
        logging.warning(
            'Failed to extract feature {}: key not found'.format(key))
      except Exception as e:
        logging.warning('Failed to extract feature {}: {}'.format(key, e))
      finally:
        if key not in kvrecord and not allow_missing:
          logging.error('Failed to parse record and not allow missing')
          kvrecord = {}
          break
    if kvrecord:
      kvrecord_list.append(kvrecord)
  return kvrecord_list

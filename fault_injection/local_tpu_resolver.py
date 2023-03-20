import tensorflow as tf

class LocalTPUClusterResolver(
    tf.distribute.cluster_resolver.TPUClusterResolver):
  """LocalTPUClusterResolver."""

  def __init__(self):
    self._tpu = ""
    self.task_type = "worker"
    self.task_id = 0

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    return None

  def cluster_spec(self):
    return tf.train.ClusterSpec({})

  def get_tpu_system_metadata(self):
    return tf.tpu.experimental.TPUSystemMetadata(
        num_cores=8,
        num_hosts=1,
        num_of_cores_per_host=8,
        topology=None,
        devices=tf.config.list_logical_devices())

  def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
    return {"TPU": 8}


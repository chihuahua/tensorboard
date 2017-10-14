
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains logic related to laying out the custom scalars dashboard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorboard.plugins.custom_scalar import metadata

def set_layout(writer, scalars_layout):
  """Sets a certain layout for the custom scalars dashboard. Writes it to disk.

  Specifically, this function will store a string representation of the layout
  within a TensorBoard summary used by the custom scalars plugin.

  When users navigate to the custom scalars dashboard, they will see a layout
  based on the proto provided to this function.

  Args:
    writer: A summary writer. Writes the summary containing the layout to disk.
    scalars_layout: The scalars_layout_pb2.Layout proto that specifies the
      layout.
  """
  tensor = tf.make_tensor_proto(
      scalars_layout.SerializeToString(), dtype=tf.string)
  summary_metadata = tf.SummaryMetadata(
      plugin_data=tf.SummaryMetadata.PluginData(
          plugin_name=metadata.PLUGIN_NAME))
  summary = tf.Summary()
  summary.value.add(tag=metadata.CONFIG_SUMMARY_TAG,
                    metadata=summary_metadata,
                    tensor=tensor)
  writer.add_summary(summary)

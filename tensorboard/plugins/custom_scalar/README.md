# TensorBoard Custom Scalars Dashboard

The TensorBoard Custom Scalars Dashboard displays scalar charts and
(unlike the Scalars Dashboard) lets users customize the layout.

Users can specify the (collapsible) categories and the charts within
each category. Furthermore, users can specify multiple tags per chart.

## A Basic Example


```
from tensorboard.plugins.custom_scalar import layout
from tensorboard.plugins.custom_scalar import layout_pb2
import tensorflow as tf

sess = tf.Session()
writer = tf.summary.FileWriter('/tmp/custom_scalars_example')

# Write some

```

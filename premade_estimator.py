#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""A DNNClassifier for the Cancer dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, shutil

import argparse
import tensorflow as tf

import cancer_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    
    args = parser.parse_args(argv[1:])

    # Fetch the data.
    (train_x, train_y), (test_x, test_y) = cancer_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    
    # Set the checkpointing schedule to every 20 mins and retain the 50 most recent checkpoints.
    my_checkpointing_config = tf.estimator.RunConfig(save_checkpoints_secs = 20*60,
                                                     keep_checkpoint_max = 50)

    # Build a DNN.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Use 3 hidden layers of 10 nodes each.
        hidden_units=[10, 10, 10],
        # Use ELU as activation function for every layer.
        activation_fn=tf.keras.activations.elu,
        # Use Adam optimizer.
        optimizer='Adam',
        # Use sum as loss reduction strategy.
        loss_reduction=tf.losses.Reduction.SUM,
        # Allow the model to choose between 2 labels.
        n_classes=2,
        # Specify directory that stores the model.
        model_dir='./model/',
        # Set checkpointing configurations.
        config=my_checkpointing_config)
                   
    # Train the model.
    """classifier.train(
            input_fn=lambda:cancer_data.train_input_fn(train_x, train_y,
                                                     args.batch_size),
            steps=args.train_steps)"""

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:cancer_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))
            
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    """expected = ['M', 'M', 'B']
    predict_x = {
        'radius_mean':[16.6, 20.6, 7.76],
        'texture_mean':[28.08, 29.33, 24.54],
        'perimeter_mean':[108.3, 140.1, 47.92],
        'area_mean':[858.1, 1265, 181],
        'smoothness_mean':[0.08455, 0.1178, 0.05263], 
        'compactness_mean':[0.1023, 0.277, 0.04362],
        'concavity_mean':[0.09251, 0.3514, 0],
        'concave_points_mean':[0.05302, 0.152, 0],
        'symmetry_mean':[0.159, 0.2397, 0.1587],
        'fractal_dimension_mean':[0.05648, 0.07016, 0.05884],
        'radius_se':[0.4564, 0.726, 0.3857],
        'texture_se':[1.075, 1.595, 1.428],
        'perimeter_se':[3.425, 5.772, 2.548],
        'area_se':[48.55, 86.22, 19.15],
        'smoothness_se':[0.005903, 0.006522, 0.007189],
        'compactness_se':[0.03731, 0.06158, 0.00466],
        'concavity_se':[0.0473, 0.07117, 0],
        'concave_points_se':[0.01557, 0.01664, 0],
        'symmetry_se':[0.01318, 0.02324, 0.02676],
        'fractal_dimension_se':[0.003892, 0.006185, 0.002783],
        'radius_worst':[18.98, 25.74, 9.456],
        'texture_worst':[34.12, 39.42, 30.37],
        'perimeter_worst':[126.7, 184.6, 59.16],
        'area_worst':[1124, 1821, 268.6],
        'smoothness_worst':[0.1139, 0.165, 0.08996],
        'compactness_worst':[0.3094, 0.8681, 0.06444],
        'concavity_worst':[0.3403, 0.9387, 0],
        'concave_points_worst':[0.1418, 0.265, 0],
        'symmetry_worst':[0.2218, 0.4087, 0.2871],
        'fractal_dimension_worst':[0.0782, 0.124, 0.07039]
    }

    predictions = classifier.predict(
        input_fn=lambda:cancer_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(cancer_data.DIAGNOSIS[class_id],
                              100 * probability, expec))"""

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

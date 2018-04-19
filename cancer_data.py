import pandas as pd
import tensorflow as tf

# Specify column names of traindata.csv and testdata.csv.
CSV_COLUMN_NAMES = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
                    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se',
                    'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
                    'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
                    'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst', 'diagnosis']

# Specify possible label values.
DIAGNOSIS = ['M', 'B']

def maybe_download():
    """Specifies paths of train data and test data."""
    
    train_path = './traindata.csv'
    # test_path = './validationdata.csv'
    test_path = './testdata.csv'

    return train_path, test_path

def load_data(y_name='diagnosis'):
    """Returns the train data and test data as (train_x, train_y), (test_x, test_y)."""
    
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


def eval_input_fn(features, labels, batch_size):
    """Evaluates or predicts based on input."""
    
    features=dict(features)
    if labels is None:
        # Use only features if no labels.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset

# Specify column types of traindata.csv and testdata.csv.
CSV_TYPES = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0], [0.0], [0.0], [0.0],[0.0],
                [0.0],[0.0],[0.0],[0.0],[0.0],[0.0], [0.0], [0.0], [0.0],[0.0],
                [0.0],[0.0],[0.0],[0.0],[0.0],[0.0], [0.0], [0.0], [0.0],[0.0], [0]]

def _parse_line(line):
    
    # Decode the line into its fields.
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary.
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    
    # Separate the label from the features.
    label = features.pop('diagnosis')

    return features, label

def csv_input_fn(csv_path, batch_size):
    
    # Create a Dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset

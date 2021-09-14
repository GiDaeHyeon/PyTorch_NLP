DATA_DIR = './dataset/'
WEIGHT = 'kykim/bert-kor-base'
NUM_CLASSES = 2
MAX_LEN = 256

BATCH_SIZE = 128
NUM_WORKERS = 4

LEARNING_RATE = 1e-4
VAL_RATIO = .1
RANDOM_STATE = 1993
FREEZE = True

MAX_EPOCHS = 1000
GPUS = 4
GRADIENT_CLIP_VAL = 3.
GRADIENT_CLIP_ALGORITHM = 'norm'
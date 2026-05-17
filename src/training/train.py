# TODO: An all-purpose training loop for models.
from src.constants import LOGGER
from models.CNN import CNN

dataset = ...
model = CNN(dataset) #placeholder, should be generic in the end
epoch_number = 0

EPOCHS = 10

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    LOGGER.info(f"EPOCH {epoch_number + 1}:")
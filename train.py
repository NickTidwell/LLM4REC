from dataloaders.datasets import PreProcessDataset, get_batch_similarity, get_movie_mapping, get_user_mapping, MovieLensDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from bivaecf.dataset import Dataset
from bivaecf.recom_bivaecf import BiVAECF
import torch
from models.llm4rec import LLM4REC
from models.projection import MLPProjection
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch.optim as optim
LATENT_DIM = 50
ENCODER_DIMS = [100]
ACT_FUNC = "tanh"
LIKELIHOOD = "pois"
NUM_EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED=10

TRAIN_AUTOENCODER=False
LOAD_AUTOENCODER="./"

DO_TRAIN = True
DO_EVAL = True


class Trainer:
    def __init__(self, model, encoder_model, dataset, user_map, movie_map, config):
        self.model = model
        self.config = config
        self.dataset = dataset
        self.encoder_model = encoder_model
        self.user_map = user_map
        self.movie_map = movie_map
        self.projection_model = MLPProjection(input_size, hidden_size, num_classes)

        self.llm_model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        system_command="test"
        user_query="test"
        # Freeze the parameters of the LLM and recommendation encoder
        for param in self.llm_model.parameters():
            param.requires_grad = False
        for param in self.encoder_model.parameters():
            param.requires_grad = False
        self.num_epochs = 1
         # Define loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.projection_model.parameters(), lr=0.001)
    def train(self):
        for epoch in range(self.num_epochs):
            for data in dataset:
                encoder_features = get_batch_similarity(data, self.encoder_model, self.movie_map, self.user_map)
                # x = model(encoder_features)
                user_features_aligned = self.projection_model(encoder_features)

                # Encode user query and concatenate with aligned user features
                encoded_user_query = self.tokenizer.encode(self.system_command, self.user_query, add_special_tokens=True, max_length=512, truncation=True, return_tensors="pt")
                combined_tensor = torch.cat((encoded_user_query, user_features_aligned), dim=1).long()
                # Process inputs through LLM
                llm_outputs = self.llm_model(combined_tensor)

                # Compute loss
                loss = None ## TODO determine what type of loss function we can use
                
                # loss = self.criterion(llm_outputs, target)

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                # Print training statistics
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item()}')
if __name__ == "__main__":
    input_size = 784  # Example input size
    hidden_size = 128
    num_classes = 10
    test_split = 0.2

    # Create your dataset and DataLoader
    dataset = PreProcessDataset("./data/")
    user_map = get_user_mapping(dataset)
    movie_map = get_movie_mapping(dataset)
    train, test = train_test_split(dataset, test_size=test_split)
    train_dataset = MovieLensDataset(train) 
    test_dataset = MovieLensDataset(test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # LOAD_OR_TRAIN_AUTOENCODER
    bivae = None
    if TRAIN_AUTOENCODER:
        train_set = Dataset.from_uir(dataset.itertuples(index=False), seed=SEED)
        bivae = BiVAECF(
        k=LATENT_DIM,
        encoder_structure=ENCODER_DIMS,
        act_fn=ACT_FUNC,
        likelihood=LIKELIHOOD,
        n_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        use_gpu=torch.cuda.is_available(),
        verbose=True
        )
        bivae.fit(train_set)
    elif LOAD_AUTOENCODER:
        print("TODO::: loading autoencoder..")


    # Instantiate your model
    model = LLM4REC(input_size, hidden_size, num_classes)

    if DO_TRAIN:
        Trainer(model, bivae, train_dataset, user_map, movie_map, None)
    if DO_EVAL
        pass #TODO:: Eval Loop
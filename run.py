from src import config
from src.logger import Logger
from src.utils import prepare_dataset
from src.dataset import MinorityDataset, CompleteDataset
from src.vae import VAE
from src.gan import SNGANHL
from src.classifier import Classifier

prepare_dataset('yeast1.dat')

clf = Classifier('clf')
gan = SNGANHL()
vae = VAE()

vae.train(MinorityDataset())
gan.train(MinorityDataset())
clf.egd_train(
    vae.encoder, gan.generator, gan.discriminator,
    CompleteDataset(), CompleteDataset(training=False), MinorityDataset()
)

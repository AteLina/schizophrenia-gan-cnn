#need to train and generate 
#1) Traditional Augmentation 
#2) DCGAN 
#3) StyleGAN 

from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 


#path object for orignal data
from pathlib import Path
real_data_path = Path("data/real_data")

from torchvision.utils import save_image


import subprocess


real_images = 'placeholder'



#DCGAN Class 
class DCGAN_Discriminator(nn.Module):
    def __init__(self, channels = 1): #channels = 1 for greyscale MRI images
        super().__init__() 
        self.conv1 = nn.Conv2d(channels, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv4 = nn.Conv2d(256, 1, 4, 1, 0)
        

    def forward(self, x):
        #define forward pass here
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.conv4(x)
        return x

class DCGAN_Generator(nn.Module):
    def __init__(self, noise_dim = 128, channels = 1): 

        super().__init__() 
        self.conv1 = nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0)
        self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv4 = nn.ConvTranspose2d(128, channels, 4, 2, 1)

    def forward(self, x):

        #need to reshape the noise vector so we can use in conv1
        x = x.view(x.size(0), -1, 1, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        #we use tanh so we can normalize pixel values between -1 and 1 
        x = torch.tanh(self.conv4(x))
        return x
    



    #---------_#


def dcgan_train():

    #making sure proper checkpoint directory exists
    Path("checkpoints/dcgan").mkdir(parents=True, exist_ok=True)

    #first, we define hyperparameters, initailzie models, and define optimizers and loss criteria 
    num_epochs = 200 
    batch_size = 16
    noise_dim = 128

    loss_criteria = nn.BCEWithLogitsLoss()

    G = DCGAN_Generator()
    D = DCGAN_Discriminator()

    opt_g = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for batch in dataloader:
            #-----------------_#
            #Train Discriminator 
            opt_d.zero_grad()
            #Discriminator Loss 

            #real -> make sure we can correctly label real images 
            labels_real = torch.ones(batch_size) 
            d_predictions = D(real_images)
            loss_d_real = loss_criteria(d_predictions, labels_real) 

            #fake -> make sure we can correctly label fake images
            #.         feed Generator noise to have it create fake images, then make sue Discriminator accurately labels fake images as false
            noise = torch.randn(batch_size, noise_dim)
            fake_images = G(noise)  
            labels_fake = torch.zeros(batch_size)
            d_predictions_fake = D(fake_images) 
            loss_d_fake = loss_criteria(d_predictions_fake, labels_fake)    


            loss_d = loss_d_real + loss_d_fake 
            loss_d.backward()
            opt_d.step()



         #------------------#
            #Train Generator -> goal is to trick Discriminator
            opt_g.zero_grad()
            noise = torch.randn(batch_size, noise_dim)
            fake_images = G(noise)
            d_predictions = D(fake_images)
            #goal = have Discriminator label fake images as real (1)
            labels_generator = torch.ones(batch_size) 

            loss_g = loss_criteria(d_predictions, labels_generator)
            loss_g.backward()
            opt_g.step()     


        #after training, we will save Generator weights so that we do not have to retrain every time. 

    torch.save(G.state_dict(), "checkpoints/dcgan_generator_saved.pt")

    return G 


#num_images tells us how many "fake" images we want the dcgan to generate, by default. 
def dcgan_generate(G, num_images = 1000):
    G.eval()  #dcga_generate is just generating images, we don't want to update Generator weights


    noise = torch.randn(num_images, 128)
    
    with torch.no_grad(): #we can save memory by telling Pytorch not to store gradients during the image generation
        fake_images = G(noise)

        for i, image in enumerate(fake_images):
            save_image(image, f"data/augmented/dcGAN/image_{i}.png")

    


def stylegan_train(): 
    #convention for using StyleGAN2-ADA. Subprocess allows us to run StyleGAN2 training from command line. 
    #we are using starting weights from pretrained FFHQ model.

    #ensuring proper checkpoints directory exists
    Path("checkpoints/stylegan").mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "python", "stylegan2-ada-pytorch/train.py",
        "--outdir=checkpoints/stylegan",
        "--data=data/real_data",
        "--resume=checkpoints/stylegan/ffhq.pkl",
        "--kimg=1000",
    ])


def stylegan_generate(num_images = 1000): 
      
      #storing stylegan_generated photos to data/augmented/stylegan directory
      subprocess.run([
        "python", "stylegan2-ada-pytorch/generate.py",
        "--outdir=data/augmented/stylegan",
        f"--seeds=0-{num_images}",
        "--network=checkpoints/stylegan/ffhq.pkl",
    ])
    

def traditional_augment(): 
    #for traditional augmentation, we will use combination of 3 transformations: rotation, zoom, and brightness and contrast
    #we will exactly double the amount of data, as we are applying the transform once to each image in real_images. 
    
    transform = transforms.Compose([
        transforms.RandomRotation(15), 
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2)])    
    

    #here, we are essentially doubling the amount of data by applying transform once to every image in real_images
    for i, image in enumerate(real_images):
        image_augmented = transform(image)
        save_image(image_augmented, f"data/augmented/traditional/image_{i}.png")




def main_augment(): 
    
    #main_augment() runs DCGAN and Style GAN training and generation + traditional augmentation.

    #making sure proper directories exist
    Path("data/augmented/dcgan").mkdir(parents=True, exist_ok=True)
    Path("data/augmented/stylegan").mkdir(parents=True, exist_ok=True)
    Path("data/augmented/traditional").mkdir(parents=True, exist_ok=True)

    #DCGAN Training + Generation

    #if we have already traiend the Generator and have the weights saved, we just load the weights and skip retraining. 
    #Otherwise, we run dcgan_train()
    if Path("checkpoints/dcgan_generator_saved.pt").exists():
        G = DCGAN_Generator()
        G.load_state_dict(torch.load("checkpoints/dcgan_generator_saved.pt"))
    else:
        G = dcgan_train()
    
    dcgan_generate(G)

    #StyleGAN Training + Generation 
    stylegan_train()
    stylegan_generate()

    #Traditional Augmentation
    traditional_augment()
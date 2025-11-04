import gc

import torch
import torch.utils.data
from dataset import *
from matplotlib import pyplot as plt
from model import *
from setup import *

if __name__ == '__main__':
    
    setup = Setup()
    device = setup.DEVICE
    
    print('Loading dataset...')
    train_logo_paths, val_logo_paths, train_clean_paths, val_clean_paths = get_paths()
    train_dataset = Dataset(train_logo_paths, train_clean_paths, patches=True)
    val_dataset = Dataset(val_logo_paths, val_clean_paths, patches=True)
    train_loader = get_data_loader(train_dataset, batch_size=setup.BATCH)
    val_loader = get_data_loader(val_dataset, batch_size=setup.BATCH)

    print('Setting up the model...')
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    criterion_mse = torch.nn.MSELoss()
    criterion_bce = torch.nn.BCELoss()

    g_optimizer = torch.optim.Adam(params=list(generator.parameters()), lr = setup.GLR)
    d_optimizer = torch.optim.Adam(params=list(discriminator.parameters()), lr = setup.DLR)

    print("Beginning training...")
    training_losses_d, training_losses_g = [], []
    val_losses_g, val_losses_d = [], []

    for epoch in range(0, setup.EPOCHS):
        training_batch_losses_d, training_batch_losses_g  = [], []
        val_batch_losses_d, val_batch_losses_g  = [], []
        
        for i, batch in enumerate(train_loader):
            
            torch.cuda.empty_cache()
            gc.collect()
            
            generator.train()
            discriminator.train()
            
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            
            logos, cleans = batch[0], batch[1]
            
            if train_dataset.patches_bool:
                logos = torch.cat(logos, dim=0).to(device)  
                cleans = torch.cat(cleans, dim=0).to(device)
            else:
                logos = logos.to(device)
                cleans = cleans.to(device)
            # logos, cleans : (BATCH*num_patches, 3, 256, 256) 
            
            d_loss_real, d_loss_fake, d_loss = 0.0, 0.0, 0.0
            g_loss_mse, g_loss_bce, g_loss = 0.0, 0.0, 0.0
            
            real_labels = torch.ones((logos.shape[0], 1)).to(device)
            fake_labels = torch.zeros((logos.shape[0], 1)).to(device)
            
            # TRAIN DISCRIMINATOR : d_loss_real + d_loss_fake
            outputs = discriminator(cleans).to(device)
            d_loss_real = criterion_bce(outputs, real_labels)

            fake_images = generator(logos).to(device)
            outputs = discriminator(fake_images.detach()).to(device)
            d_loss_fake = criterion_bce(outputs, fake_labels)

            d_loss = (d_loss_real + d_loss_fake)/2
            d_loss.backward()
            d_optimizer.step()
            
            # TRAIN GENERATOR : g_loss_mse*lambda + g_loss_bce
            g_loss_mse = criterion_mse(fake_images, cleans)
            outputs = discriminator(fake_images).to(device)
            g_loss_bce = criterion_bce(outputs, real_labels)
            
            g_loss = setup.LAMBDA*g_loss_mse + g_loss_bce
            g_loss.backward()
            g_optimizer.step()
    
            if i % 200 == 0:
                print("T_Epoch: [%d/%d], Step: [%d/%d]  |  D_R: %.3f, D_F: %.3f |  G_MSE: %.3f, G_BCE: %.3f  |  D_avg_Loss: %.3f  G_avg_Loss: %.3f " \
                      % (epoch+1, setup.EPOCHS, i, len(train_loader),d_loss_real.item(), d_loss_fake.item() ,setup.LAMBDA*g_loss_mse.item(), g_loss_bce.item() , d_loss.item(), g_loss.item())) 
            
            d_loss = d_loss.to(torch.device("cpu"))
            g_loss = g_loss.to(torch.device("cpu"))
            training_batch_losses_d.append(d_loss) 
            training_batch_losses_g.append(g_loss) 
            
            
        for i, batch in enumerate(val_loader):
            
            generator.eval()
            discriminator.eval()
            
            with torch.no_grad():
            
                logos, cleans = batch[0], batch[1]
                
                if val_dataset.patches_bool:
                    logos = torch.cat(logos, dim=0).to(device)
                    cleans = torch.cat(cleans, dim=0).to(device)
                else:
                    logos = logos.to(device)
                    cleans = cleans.to(device)
                # logos, cleans : (BATCH*num_patches, 3, 256, 256) 
                
                d_loss_real, d_loss_fake, d_loss = 0.0, 0.0, 0.0
                g_loss_mse, g_loss_bce, g_loss = 0.0, 0.0, 0.0
                
                real_labels = torch.ones((logos.shape[0], 1)).to(device)
                fake_labels = torch.zeros((logos.shape[0], 1)).to(device)
                
                # Discriminator with real images
                outputs = discriminator(cleans).to(device)
                d_loss_real = criterion_bce(outputs, real_labels)

                # Discriminator with fake images
                fake_images = generator(logos).to(device)
                outputs = discriminator(fake_images.detach()).to(device)
                d_loss_fake = criterion_bce(outputs, fake_labels)

                d_loss = (d_loss_real + d_loss_fake)/2
                
                # Generator
                g_loss_mse = criterion_mse(fake_images, cleans)
                outputs = discriminator(fake_images).to(device)
                g_loss_bce = criterion_bce(outputs, real_labels)
                
                g_loss = setup.LAMBDA*g_loss_mse + g_loss_bce

                if i % 200 == 0:
                    print("V_Epoch: [%d/%d], Step: [%d/%d], D_avg_Loss: %.3f,  G_avg_Loss: %.3f " % (epoch+1, setup.EPOCHS, i, len(val_loader),d_loss.item(), g_loss.item())) 
                
                d_loss = d_loss.to(torch.device("cpu"))
                g_loss = g_loss.to(torch.device("cpu"))
                val_batch_losses_d.append(d_loss) 
                val_batch_losses_g.append(g_loss) 
            
        

        # get the average results for each epoch for training
        training_losses_d.append(float(sum(training_batch_losses_d) / len(training_batch_losses_d)))
        training_losses_g.append(float(sum(training_batch_losses_g) / len(training_batch_losses_g)))
        
        # get the average results for each epoch for validation
        val_losses_d.append(float(sum(val_batch_losses_d) / len(val_batch_losses_d)))
        val_losses_g.append(float(sum(val_batch_losses_g) / len(val_batch_losses_g)))

        # save model after every epoch
        torch.save(generator.state_dict(), f"checkpoints/G-B{setup.BATCH}-G-{setup.GLR}-D-{setup.DLR}-{setup.LAMBDA}MSE-E{epoch+1}.pt")
        torch.save(discriminator.state_dict(), f"checkpoints/D-B{setup.BATCH}-G-{setup.GLR}-D-{setup.DLR}-{setup.LAMBDA}MSE-E{epoch+1}.pt")


    plt.subplot(1,2,1)
    plt.plot(training_losses_d)
    plt.plot(training_losses_g)
    plt.title('Losses vs Epochs (Train)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Discriminator', 'Generator'])
    
    plt.subplot(1,2,2)
    plt.plot(val_losses_d)
    plt.plot(val_losses_g)
    plt.title('Losses vs Epochs (Val)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Discriminator', 'Generator'])
    
    plt.savefig(f'plots/GD-B{setup.BATCH}-G-{setup.GLR}-D-{setup.DLR}-{setup.LAMBDA}MSE-E{setup.EPOCHS}.jpg')



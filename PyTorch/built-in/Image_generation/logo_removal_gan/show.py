
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from dataset_file import *
from model import *
from setup import * 


if __name__ == '__main__':

    setup = Setup()
        
    train_logo_paths, val_logo_paths, train_clean_paths, val_clean_paths = get_paths()
    val_dataset = Dataset(val_logo_paths, val_clean_paths, patches=False)
    val_loader = get_data_loader(val_dataset, batch_size=setup.BATCH_show)

    logos, cleans = next(iter(val_loader))
    
    if val_dataset.patches_bool:
        logos = torch.cat(logos, dim=0)
        cleans = torch.cat(cleans, dim=0)
        
    generator = Generator()
    generator.eval()
    
    try:
        if setup.AUTO == True:   
            generator.load_state_dict(torch.load(f"checkpoints/AUTOG-B{setup.BATCH}-G-{setup.GLR}-E{setup.EPOCHS}.pt"))
        else:   
            generator.load_state_dict(torch.load(f"checkpoints/G-B{setup.BATCH}-G-{setup.GLR}-D-{setup.DLR}-{setup.LAMBDA}MSE-E{setup.EPOCHS}.pt"))
            
        generated = generator.forward(logos)
        
        for logo, gen, clean in zip(logos, generated, cleans):
            logo = denormalize(logo)
            gen = denormalize(gen)
            clean = denormalize(clean)

            _, ax = plt.subplots(1,3, figsize=(20,10))
            ax[0].imshow(logo)
            ax[0].title.set_text('Logo')
            ax[1].imshow(gen)
            ax[1].title.set_text('Generated')
            ax[2].imshow(clean)
            ax[2].title.set_text('Clean')
            plt.show()
            plt.pause(1)
            
    except:
        print('Such checkpoint for given parameters was not found.')



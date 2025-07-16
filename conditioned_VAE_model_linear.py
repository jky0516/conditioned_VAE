import torch
from torch import nn
from torch.nn import functional as F


class ConditionedVAE(nn.Module):


    def __init__(self, in_channels=1, latent_dim=256,
                cond_dim=4, emb_channels=16, hidden_dims=None):
        
        super().__init__()

        self.latent_dim = latent_dim
        """
        Embedding the conditioned input.
        """
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, emb_channels)
        )

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(256 * 3 * 2, latent_dim) 
        self.fc_var = nn.Linear(256 * 3 * 2, latent_dim)

        self.decoder_input = nn.Linear(latent_dim + emb_channels, 256 * 3 * 2)

        # Build Decoder
        modules = []    
          
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Upsample(size=(35, 31)),
                            nn.Conv2d(hidden_dims[-1], out_channels=1,
                                      kernel_size= 3, padding= 1),
                            nn.LeakyReLU()
                            )

        self.decoder = nn.Sequential(*modules)


    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(x)

        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z_cat):
        """
        Maps the given latent codes
        onto the image space.
        """
        result = self.decoder_input(z_cat)
        result = result.view(-1, 256, 3, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, cond_input): 

        B = cond_input.size(0)

        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)

        cond_emb = self.cond_mlp(cond_input) 
        z_cat = torch.cat([z, cond_emb], dim=1)  

        res = self.decode(z_cat)
        res = F.softmax(res.view(B, -1), dim=1).view(B, 1, 35, 31)

        return res, mu, log_var
    
    def sample(self, cond_input):

        B = cond_input.size(0)

        z = torch.rand(B, self.latent_dim)
        cond_emb = self.cond_mlp(cond_input)
        z_cat = torch.cat([z, cond_emb], dim=1)

        samples = self.decode(z_cat)
        samples= F.softmax(samples.view(B, -1), dim=1).view(B, 1, 35, 31)
        return samples
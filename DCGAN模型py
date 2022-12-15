# batch-optimization
def dataloader(batch_size:int, original_data:torch.tensor) -> list:
    N = original_data.shape[0]
    alist = [i for i in range(N)]
    result = list()
    np.random.shuffle(alist)
    for i in range(N//batch_size):
        if (i+1)*batch_size >= N:
            result.append(original_data[i*batch_size:])
        else:
            result.append(original_data[i*batch_size:(i+1)*batch_size])
    return result
    
    
# DCGAN要做初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        
# 畫圖函式
def Draw(array):
    plt.imshow((array*0.5 + 0.5).permute(1,2,0).detach().to(device = 'cpu'))
    plt.axis('off')
def multi_Draw(Input):
    flat = 1
    for obj in Input:
        plt.subplot(1, 5, flag)
        Draw(obj.to(device = 'cpu'))
        flag += 1
    plt.show()
        
# Generator範例
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__();
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_channels = nz,
                                                      out_channels = ngf * 8,
                                                      kernel_size = 4,
                                                      stride = 1,
                                                      padding = 0,
                                                      bias = False
                                                     ),
                                   nn.BatchNorm2d(ngf * 8),
                                   nn.ReLU()
                                   )
        ##### (ngf*8) x 4 x 4 #####
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(in_channels = ngf * 8,
                                                      out_channels = ngf * 4,
                                                      kernel_size = 4,
                                                      stride = 2,
                                                      padding = 1,
                                                      bias = False
                                                     ),
                                   nn.BatchNorm2d(ngf * 4),
                                   nn.ReLU()
                                   )
        ##### (ngf*4) x 8 x 8 #####
        self.conv3 = nn.Sequential(nn.ConvTranspose2d(in_channels = ngf * 4,
                                                      out_channels = ngf * 2,
                                                      kernel_size = 4,
                                                      stride = 2,
                                                      padding = 1,
                                                      bias = False
                                                     ),
                                   nn.BatchNorm2d(ngf * 2),
                                   nn.ReLU()
                                   )
        ##### (ngf*2) x 16 x 16 #####
        self.conv4 = nn.Sequential(nn.ConvTranspose2d(in_channels = ngf * 2,
                                                      out_channels = ngf,
                                                      kernel_size = 4,
                                                      stride = 2,
                                                      padding = 1,
                                                      bias = False
                                                     ),
                                   nn.BatchNorm2d(ngf),
                                   nn.ReLU()
                                   )
        ##### ngf x 16 x 16 #####
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(in_channels = ngf,
                                                      out_channels = nc,
                                                      kernel_size = 4,
                                                      stride = 2,
                                                      padding = 1,
                                                      bias = False
                                                     ),
                                   nn.Tanh()
                                   )
    def forward(self, x):
        X = self.conv1(x)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)
        X = self.conv5(X)
        return X
netG = Generator().to(device = 'cuda')
netG.apply(weights_init)

# Discriminator範例
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__();
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = nc,
                                             out_channels = ndf,
                                             kernel_size = 4,
                                             stride = 2,
                                             padding = 1,
                                             bias = False
                                            ),
                                   nn.LeakyReLU(0.2, inplace = True)
                                  )
        ##### ndf x 32 x 32 #####
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = ndf,
                                             out_channels = ndf*2,
                                             kernel_size = 4,
                                             stride = 2,
                                             padding = 1,
                                             bias = False
                                            ),
                                   nn.BatchNorm2d(ndf*2),
                                   nn.LeakyReLU(0.2, inplace = True)
                                  )
        ##### (ndf*2) x 16 x 16 #####
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = ndf*2,
                                             out_channels = ndf*4,
                                             kernel_size = 4,
                                             stride = 2,
                                             padding = 1,
                                             bias = False
                                            ),
                                   nn.BatchNorm2d(ndf*4),
                                   nn.LeakyReLU(0.2, inplace = True)
                                  )
        ##### (ndf*4) x 8 x 8 #####
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels = ndf*4,
                                             out_channels = ndf*8,
                                             kernel_size = 4,
                                             stride = 2,
                                             padding = 1,
                                             bias = False
                                            ),
                                   nn.BatchNorm2d(ndf*8),
                                   nn.LeakyReLU(0.2, inplace = True)
                                  )
        ##### (ndf*8) x 4 x 4 #####
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels = ndf*8,
                                             out_channels = 1,
                                             kernel_size = 4,
                                             stride = 1,
                                             padding = 0,
                                             bias = False
                                            ),
                                   nn.Sigmoid()
                                  )
    def forward(self, x):
        X = self.conv1(x)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)
        X = self.conv5(X)
        return X
netD = Discriminator().to(device = 'cuda')
netD.apply(weights_init)


# 最佳化器範例
criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr = 0.0001, betas = (0.5, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr = 0.0001, betas = (0.5, 0.999))


# 模型訓練
import warnings
warnings.filterwarnings("ignore")
X = X.to(device = 'cuda')
LOSSES = list()
photo_show = torch.randn(5, nz, 1, 1).detach().to(device = 'cuda')
batch_size = 32
for epoch in range(1000):
    for real_data in dataloader(batch_size, X):
        ##### 辨別器更新 #####
        optimizerD.zero_grad()
        label = torch.tensor(np.ones(shape = (real_data.shape[0], ))).to(torch.float32).to(device = 'cuda')
        output = netD(real_data).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        noise = torch.randn(real_data.shape[0], nz, 1, 1).to(device = 'cuda')
        fake = netG(noise)
        FAKE = torch.tensor(np.zeros(shape = (real_data.shape[0], ))).to(torch.float32).to(device = 'cuda')
        errD_fake = criterion(netD(fake.detach()).view(-1), FAKE)
        errD_fake.backward()
        optimizerD.step()

        ##### 生成器更新 #####
        optimizerG.zero_grad()
        noise.data.copy_(torch.randn(real_data.shape[0], nz, 1, 1)).to(device = 'cuda')
        fake = netG(noise)
        errG = criterion(netD(fake).view(-1), 1 - FAKE)
        errG.backward()
        optimizerG.step()
       
    print('Epoch:', epoch + 1)
    multi_Draw(netG(photo_show))

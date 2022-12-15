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

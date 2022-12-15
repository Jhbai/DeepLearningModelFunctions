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

import torch

# follow MoCo memory bank implementation
class MemoryBank():
    def __init__(self, size = 128, dim = 128):
        self.size = size
        self.feature = torch.randn(size, dim, dtype=torch.bfloat16)
        self.queue_ptr = torch.zeros(1, dtype=torch.long)
        self.K = size
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.feature[ptr : ptr + batch_size, :] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
        
    def update(self, keys):
        self._dequeue_and_enqueue(keys)
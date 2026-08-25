import sys
try:
    import torch
    print('torch', torch.__version__)
    print('CUDA:', torch.cuda.is_available())
    print('MPS:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)
except Exception as e:
    print('ERR', type(e).__name__, e)
    sys.exit(0)

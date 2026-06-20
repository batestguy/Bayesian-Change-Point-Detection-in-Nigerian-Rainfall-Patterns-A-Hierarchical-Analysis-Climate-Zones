try:
    import pymc as pm
    print(f"PyMC {pm.__version__} already installed")
except ImportError:
    print("NEED_INSTALL")

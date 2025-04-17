import models_cifar

dependencies = ['torch']

models = filter(lambda name: name.startswith("cifar"), dir(models_cifar))
globals().update({model: getattr(models_cifar, model) for model in models})
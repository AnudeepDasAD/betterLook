"""
A simple example that demonstrates how to run a single attack against
a PyTorch ResNet-18 model for different epsilons and how to then report
the robust accuracy.
"""
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, LinfFastGradientAttack
import torch
# from betterLook.experimentResnet import *
# from cbcl_test import assembleCentroids, CentroidConfig
from attnBartDualNew import *
import gc
from foolbox.criteria import TargetedMisclassification


'''PGD attack'''
def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # instantiate a model (could also be a TensorFlow or JAX model)

    # weights=models.ResNet50_Weights.IMAGENET1K_V2
    # model = models.resnet50(weights=weights)
    # model.eval()

    # model = prepareModel('resnet50+madry').eval()
    # model.eval()
    model = torch.load('resnet_glance_dualall_imgnette_attn_bartrand_trial1_epoch9.pt').cpu()

    # total_centroid_paths = ['centroids_imagenette.pt']
    # total_centroids = assembleCentroids(total_centroid_paths)
    # model = ConvLstm(total_centroids)
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model.cuda(), bounds=(-1000, 1000))

    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    # images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    # images = images.reshape((16, 3, 224, 224))
    # # ds = torch.load('imagenette_preproc_ensadvincres_valset.pt')
    # ds = torch.load('imagenette_preproc_ensadvincres_valset.pt')
    # dl = torch.utils.data.DataLoader(ds, batch_size = 8, shuffle=False)
    _, dl, ds = prepareDatasets()
    x_advs = []
    ts = []
    num_batches_processed = 0

    attack = LinfPGD()
    epsilons = [
        8/255
    ]
    for images, labels in dl:
        # if num_batches_processed < 49:
        #     num_batches_processed += 1
        #     continue
        images = images.to(device)
        labels = labels.to(device)
        # labels = torch.tensor([ds.classes[val] for val in labels]).to(device)
        # images.requires_grad=True
        criterion = TargetedMisclassification(labels) # NEW !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        clean_acc = accuracy(fmodel, images, labels)
        print(f"clean accuracy:  {clean_acc * 100:.1f} %")
        
        raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons, criterion=criterion)
        
        print("robust accuracy for perturbations with")
        robust_accuracy = 1 - torch.mean(success, axis=-1, dtype=torch.float32)
        for eps, acc in zip(epsilons, robust_accuracy):
            print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

        # print(len(clipped_advs[0])) # 8

        x_advs.append(clipped_advs[0])
        ts.append(labels.to('cpu'))

        num_batches_processed += 1
        print(f'num batches processed: {num_batches_processed}')
        if num_batches_processed % 80 == 0 or num_batches_processed == 491: #625 for imagenet100, 491 for imagenette
            torch.save([x_advs, ts], f'x_advs_attn_bs8_dualnew2_imagenette_fool_pgd_eps8255_dec22{num_batches_processed}.pt')
            x_advs = []
            ts = []
            gc.collect()



    # # calculate and report the robust accuracy (the accuracy of the model when
    # # it is attacked)
    # robust_accuracy = 1 - success.float32().mean(axis=-1)
    # print("robust accuracy for perturbations with")
    # for eps, acc in zip(epsilons, robust_accuracy):
    #     print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

    # # we can also manually check this
    # # we will use the clipped advs instead of the raw advs, otherwise
    # # we would need to check if the perturbation sizes are actually
    # # within the specified epsilon bound
    # print()
    # print("we can also manually check this:")
    # print()
    # print("robust accuracy for perturbations with")
    # for eps, advs_ in zip(epsilons, clipped_advs):
    #     acc2 = accuracy(fmodel, advs_, labels)
    #     print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
    #     print("    perturbation sizes:")
    #     perturbation_sizes = (advs_ - images).norms.linf(axis=(1, 2, 3)).numpy()
    #     print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
    #     if acc2 == 0:
    #         break


if __name__ == "__main__":
    main()
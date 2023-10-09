for i in 0.2 0.3 0.4 0.5
do
    python main.py --r $i --desc "pxy_cifar10_gmm_lpri_"$i;
done

# for i in 0.4 0.5
# do
#     python main.py --r $i --root "/media/hdd/fb/cifar-100-python/" --desc "pxy_cifar100_gmm_lpri"
# done
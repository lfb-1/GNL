# CIFAR10
# for s in 42 123 678
# do
#     for j in pxy pyx
#     do
#         for i in 0.2 0.3 0.4 0.5
#         do
#             python main.py --r $i --desc "c10"$j"_softmax_"$i"_"$s --optim_goal $j --config cifar10 --seed $s;
#         done
#     done
# done

# CIFAR100
for s in 42 123 678
do
    for j in pxy pyx
    do
        for i in 0.2 0.3 0.4 0.5
        do
            python main.py --r $i --root /media/hdd/fb/cifar-100-python --desc "c100"$j"_normgen_"$i"_"$s --optim_goal $j --config cifar100 --seed $s;
        done
    done
done


# for i in 0.2 0.4 0.6 0.8
# do
#     python main.py --r $i --config red --desc "pyx_red_"$i --root /run/media/Data/red_blue/ --optim_goal pyx
# done
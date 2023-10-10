for i in 0.2 0.3 0.4 0.5
do
    # python main.py --r $i --desc "c10_pxy_normgen_"$i --optim_goal "pxy";
    python main.py --r $i --desc "c10_pyx_normgen_"$i --optim_goal pyx;
done

for i in 0.2 0.3 0.4 0.5
do
    python main.py --r $i --root "/media/hdd/fb/cifar-100-python/" --desc "c100_pxy_normgen_"$i --optim_goal pxy;
    python main.py --r $i --root "/media/hdd/fb/cifar-100-python/" --desc "c100_pyx_normgen_"$i --optim_goal pyx;

done

# for i in 0.2 0.4 0.6 0.8
# do
#     python main.py --r $i --config red --desc "pyx_red_"$i --root /run/media/Data/red_blue/ --optim_goal pyx
# done
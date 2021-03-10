mkdir ./checkpoints/importance_sampling125
python tmp.py --name importance_sampling125 --sample_p .125 --niter 10 --niter_decay 0 --lr 0.00001 --phase test_sampler --which_model_netG siggraphlocal --samp importance 

mkdir ./checkpoints/importance_sampling031
python tmp.py --name importance_sampling031 --sample_p .031 --niter 10 --niter_decay 0 --lr 0.00001 --phase test_sampler --which_model_netG siggraphlocal --samp importance

mkdir ./checkpoints/normal_sampling125
python tmp.py --name normal_sampling125 --sample_p .125 --niter 10 --niter_decay 0 --lr 0.00001 --phase test_sampler --which_model_netG siggraphlocal --samp normal 

mkdir ./checkpoints/normal_sampling031
python tmp.py --name normal_sampling031 --sample_p .031 --niter 10 --niter_decay 0 --lr 0.00001 --phase test_sampler --which_model_netG siggraphlocal --samp normal

mkdir ./checkpoints/uniform_sampling125
python tmp.py --name uniform_sampling125 --sample_p .125 --niter 10 --niter_decay 0 --lr 0.00001 --phase test_sampler --which_model_netG siggraphlocal --samp uniform 

mkdir ./checkpoints/uniform_sampling031
python tmp.py --name uniform_sampling031 --sample_p .031 --niter 10 --niter_decay 0 --lr 0.00001 --phase test_sampler --which_model_netG siggraphlocal --samp uniform
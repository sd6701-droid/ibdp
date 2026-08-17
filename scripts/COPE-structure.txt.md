(base) samprasmanueldsouza@b-10-27-70-44 ~ % ssh sd6701@login.torch.hpc.nyu.edu
(base) samprasmanueldsouza@b-10-27-70-44 ~ % ssh sd6701@bigpurple.nyumc.org
sd6701@bigpurple.nyumc.org's password: 
Last login: Wed Aug  5 12:02:16 2026 from 10.128.210.195
   __  __  __  __                   _    __  _           __         __ 
  / / / / / / / /_  _____  _____   | |  / / (_) ____    / / ___    / /_
 / / / / / / / __/ / ___/ / __  |  | | / / / / / __ \  / / / _ \  / __/
/ /_/ / / / / /_  / /    / /_/  |  | |/ / / / / /_/ / / / /  __/ / /_  
\____/ /_/  \__/ /_/     \____/\_\ |___/ /_/  \____/ /_/  \___/  \__/  
                                           NYU Langone Health HPC      

Use the following commands to adjust your environment:
'module avail'            - show available modules
'module add <module>'     - adds a module to your environment for this session
'module initadd <module>' - configure module to be loaded at every login
   
    BigPurple User Guide available at: http://bigpurple-ws.nyumc.org/wiki
    New HPC Portal: https://hpcmed.org/
    You may email <hpc_admins@nyumc.org> for any further assistance.

PLEASE NOTE:
Compute intensive applications and programs such as python and R should not be
run on the login nodes as it affects performance for all other users.
These processes will be automatically killed.
################################################################################
To submit a ticket, please log in to the Ivanti portal.
Detailed instructions and a short user guide will be available on our HPC 
documentation page before the transition date.

https://nyu-amc.ivanticloud.com/service/

For additional assistance and guidance please free free to explore our clusters
dedicated help page. 

https://bigpurple-ws.nyumc.org/wiki/index.php/BigPurple_Help

**** We are implementing a 10TB user quota on scratch space, soon. CLEAN UP! ****
################################################################################

Try OnDemand - Graphical interface to Jupyter, RStudio, Virtual Desktop:

https://ondemand.hpc.nyumc.org

Thank you for your understanding and continued use of our HPC resources.


Home Block Quota: 56.87G/100G
Home File Quota: 410234/40000 Warning: Over Quota, Grace Period is 56 days 

For Quota Help Please See:
	http://bigpurple-ws.nyumc.org/wiki/index.php/Data-Management#Exceeding_Home_Directory_quotas

Loading default-environment
  Loading requirement: json-c/0.17 slurm/current standard-tools/1.0
Loading gcc/15.2.0
  Loading requirement: mpfr/4.0.1 mpc/1.1.0 gmp/6.1.2
(base) [sd6701@bigpurple-ln2 ~]$ D=/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential
(base) [sd6701@bigpurple-ln2 ~]$ 
(base) [sd6701@bigpurple-ln2 ~]$ # 1. top level
(base) [sd6701@bigpurple-ln2 ~]$ ls -la "$D"
total 114
drwxr-xr-x   9 ys1001 shenlab  4096 Apr  1 11:09 .
drwxr-sr-x   4 ys1001 shenlab  4096 Mar  2 15:48 ..
drwxr-xr-x 157 ys1001 shenlab  8192 Feb 12 16:24 12_Months
drwxr-xr-x  14 ys1001 shenlab  4096 Feb 11 18:47 42_Months
drwxr-xr-x 159 ys1001 shenlab  8192 Feb 12 16:25 6_Months
drwxr-xr-x   2 ys1001 shenlab 16384 Feb 11 15:08 Arm_Restraint_6mo
-rwxr-xr-x   1 ys1001 shenlab  6148 Feb 19 08:14 .DS_Store
drwxr-xr-x   2 ys1001 ys1001   4096 Apr  1 11:09 final-datasets
drwxr-xr-x   2 ys1001 shenlab  8192 Feb 11 15:48 Freeplay_6mo
drwxr-xr-x   3 ys1001 shenlab  8192 Feb 11 12:08 MAAP_Cropped_Videos
(base) [sd6701@bigpurple-ln2 ~]$ 
(base) [sd6701@bigpurple-ln2 ~]$ # 2. folder tree, two levels deep
(base) [sd6701@bigpurple-ln2 ~]$ find "$D" -maxdepth 2 -type d | sort
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/100
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/104
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/107
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/109
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/111
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/119
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/126
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/127
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/142
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/144
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/153
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/168
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/170
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/174
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/18
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/19
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/193
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/20
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/201
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/219
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/225
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/226
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/237
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/249
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/25
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/279
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/311
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/328
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/335
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/336
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/350
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/363
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/372
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/378
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/38
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/385
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/393
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/406
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/421
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/423
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/424
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/434
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/441
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/45
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/452
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/464
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/468
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/47
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/471
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/5
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/501
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/516
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/534
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/550
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/562
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/572
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/578
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/59
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/591
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/6
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/601
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/614
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/651
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/71
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/74
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/77
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/79
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/8
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/94
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/99
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21052
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21056
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21069
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21116
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21146
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21152
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21184
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21215
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21226
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21259
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21269
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21349
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21478
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21587
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21588
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21603
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21614
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21659
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21663
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21664
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21706
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21722
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21752
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21780
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21793
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21805
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21994
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22000
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22179
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22289
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22306
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22380
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22381
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22382
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22400
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22439
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22440
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22588
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22647
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22801
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22822
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22901
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22911
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23066
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23107
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23112
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23274
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23288
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23387
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23455
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23697
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23827
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24232
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24236
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24309
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24341
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24432
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24542
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24586
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24873
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24886
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24928
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24957
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M25179
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N194
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N208
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N231
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N410
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N419
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N507
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N564
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N643
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N801
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T1366
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T1455
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T1551
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T1768
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T1836
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T1862
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T1992
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T2004
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T2096
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T2177
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T2224
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T2274
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/AV consents
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/DELL Laptop Backup FIles
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/fNIRS
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Heart Rate
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Jumble CSVs
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Jumble PIPs
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Puzzles CSVs
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/QUILS
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/R Scripts
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Laptop Backup Files
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Test fNIRS
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/100
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/102
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/107
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/109
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/111
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/119
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/127
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/140
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/144
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/153
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/168
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/17
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/170
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/174
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/18
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/19
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/1912
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/196
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/199
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/20
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/201
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/218
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/225
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/226
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/235
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/237
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/249
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/268
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/299
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/311
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/335
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/336
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/350
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/354
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/363
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/373
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/378
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/406
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/421
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/423
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/424
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/426
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/439
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/441
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/45
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/452
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/464
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/468
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/471
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/49
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/5
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/516
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/534
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/547
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/55
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/562
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/572
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/59
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/591
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/6
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/601
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/614
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/624
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/638
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/71
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/74
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21052
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21056
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21069
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21071
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21073
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21146
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21152
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21184
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21226
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21228
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21269
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21349
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21570
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21587
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21588
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21603
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21614
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21620
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21642
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21658
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21659
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21660
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21662
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21663
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21672
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21706
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21722
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21752
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21780
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21793
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21850
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21994
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22000
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22179
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22272
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22289
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22306
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22380
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22381
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22382
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22383
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22400
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22417
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22439
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22440
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22588
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22624
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22822
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22901
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22911
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M23066
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M23112
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M23274
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M23276
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M23288
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M23366
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M23387
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M23405
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M23468
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M23635
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M23697
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24232
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24283
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24341
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24432
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24542
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24591
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24718
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24749
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24793
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24886
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24926
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24957
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M25037
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M25139
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M25139-2
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M25179
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M25401
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N186
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N194
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N208
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N231
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N367
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N419
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N507
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N564
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N643
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N801
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N850
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N853
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N863
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/Arm_Restraint_6mo
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/final-datasets
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/Freeplay_6mo
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/MAAP_Cropped_Videos
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/MAAP_Cropped_Videos/MAAP Cropped Videos Redo
(base) [sd6701@bigpurple-ln2 ~]$ 
(base) [sd6701@bigpurple-ln2 ~]$ # 3. size + file count per top-level folder
(base) [sd6701@bigpurple-ln2 ~]$ for d in "$D"/*/; do
>   printf "%8s  %5d files  %s\n" "$(du -sh "$d" 2>/dev/null | cut -f1)" \
>          "$(find "$d" -type f 2>/dev/null | wc -l)" "$d"
> done
    999G   2050 files  /gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/
    126G  16239 files  /gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/
    976G   2075 files  /gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/
     84G    140 files  /gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/Arm_Restraint_6mo/
    257K      4 files  /gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/final-datasets/
    194G    131 files  /gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/Freeplay_6mo/
     61G    157 files  /gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/MAAP_Cropped_Videos/
(base) [sd6701@bigpurple-ln2 ~]$ 
(base) [sd6701@bigpurple-ln2 ~]$ # 4. what file types exist overall
(base) [sd6701@bigpurple-ln2 ~]$ find "$D" -type f 2>/dev/null | sed 's/.*\.//' | tr 'A-Z' 'a-z' | sort | uniq -c | sort -rn | head -15
   3160 mov
   2342 json
   1495 csv
   1425 mp4
   1081 mat
   1075 hdr
   1072 zip
    957 ds_store
    924 snirf
    907 txt
    866 wl2
    866 wl1
    849 nirs
    748 tri
    429 id
(base) [sd6701@bigpurple-ln2 ~]$ 
(base) [sd6701@bigpurple-ln2 ~]$ # 5. the PDFs you suspect are there
(base) [sd6701@bigpurple-ln2 ~]$ find "$D" -iname "*.pdf" 2>/dev/null
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/fNIRS/Processing/Eric Project/Satori Processed/pp1/Model/customProbe/standard.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/fNIRS/Processing/Eric Project/Satori Processed/pp1/Model/Montagepp1/standard.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M21587/M21587_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/69/69_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/439/439_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/314/314_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/310/310_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/423/423_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/465/465_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M21015/M21015_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T2042/T2042_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M22380/M22380_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M21659/M21659_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/471/471_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/534/534_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/N801/N801_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/605/605_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/643/643_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/226/226_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T1992/T1992_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T2788/T2788_42m_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/327/327_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/311/311_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M21620/M21620_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M21618/M21618_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/176/176_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/126/126_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/535/535_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M21526/Participant ID: M21526.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/65/65_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/295/295_42_notes 2.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/295/295_42_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/312/312_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/109/109_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/372/372_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T1836/T1836_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M21410/M21410_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/548/548_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M21403/M21403_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T2974/T2974_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/N182/N182_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M21663/M21663_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/191/191_42_notes 2.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T2383/T2383_42_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T2096/T2096_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/601/601_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T3194/T3194_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/385/385_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T2605/T2605_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/45/45_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T433/T433-42-visit notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T853/T853_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/N386/N386_42_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/99/99_42_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/217/217_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M24864/Task Videos/M24864_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/142/142_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T1551/v.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M21657/M21657_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/336/336_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M24852/M24852_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/350/350_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/421/439_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M25559/M25559_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/615/625_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/540/Task Videos/540_42_visitnotes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/173/173_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/582/582_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/236/236_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/307/307_42_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T1862/T1862_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M21660/M21660_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/641/641_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M21349/M21349_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/20/20_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T4095/T4095_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/71/71_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/N179/N179_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/464/464_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T588/T588_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/M23479/M23479_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/414/414_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/452/452_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/440/440_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/557/557_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/15/15_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/174/174_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/18/18_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T2224/T2224_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/T2282/T2282_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/42_Months/Task Videos & Notes/634/634_42_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N507/N507_12_E2_Notes_Sheet.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21805/M21805_12_Month_E2_Notes_Sheet.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N208/N208_6_E1_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/423/423_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/119/119_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/107/107_12_Stool_Survey.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/516/516_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/441/12_Month_E2_Notes_Sheet-14.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/471/471_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/534/534_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N801/N801_12_E1_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21056/M21056_12_E2_Notes_Sheet.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21069/M21069_12_E2_Notes_docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/59/59_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/572/572_12_E2_Notes_Sheet.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/226/226_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T1992/T1992_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24432/M24432_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24957/M24957_12_E2_Notes .pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23112/M23112_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T1455/T1455_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/311/311_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22289/M22289_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21603/M21603_12_E1_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24873/M24873_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/6/6_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23107/M23107_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22000/M22000_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/501/501_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24586/M24586_12_E2_Notes .pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21184/M21184_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/19/19_12_E2_notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/126/126_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23827/M23827_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/378/378 _12_Microbiome_Survey.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/328/328_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/5/5_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/109/109 _12_Microbiome_Survey.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/109/109_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/372/372_E3_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/372/372_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T1836/T1836_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21614/M21614_12_E2_Notes_Sheet.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/578/578_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21269/M21269_12_E1_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21663/M21663_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21146/M21146_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23455/M23455_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22179/M22179_12_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21793/M21793_12_E2_Notes_Sheet.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/550/550_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N643/N643_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T2096/T2096_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/651/651_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/601/601_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/77/77_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/77/77_12_E3_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/385/385_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22822/M22822_12_E3_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22822/M22822_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/406/406_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N564/N564_12_E2_Notes_Sheet.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21722/M21722_12_E2_Notes_Sheet.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23066/M23066_12_E3_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23066/M23066_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/45/45_12_E2_Notes_Sheet.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/424/424_E3_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/424/424_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23274/M23274_12_E3_Notes .docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T2274/T2274 _12_Microbiome_Survey.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T2274/T2274_12_E3_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N419/N419_12_E1_Notes_Sheet.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/99/99_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/100/100_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21226/M21226_12_E2_Notes_Sheet.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T2177/T2177_12_E1_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/94/94_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/142/142 Notes Sheet E2.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21052/M21052_E2_Notes_Sheet.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T1551/T1551_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/201/201_12_E2_notes .pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/614/614_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21116/M21116_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23288/M23288_12_E2_Notes .docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/336/336 _12_Microbiome_Survey.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N194/N194_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/38/38_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21152/M21152_12_E2_Notes_Sheet.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/350/350_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/421/421_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22440/M22440_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24309/M24309_12_E2_Notes .pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/193/193_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24542/E2/M24542_12_Month_E2_Notes .pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22381/M22381_E2_Notes_Sheet.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22647/M22647_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21588/M21588_12_E2_Notes_Sheet.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/25/25_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/T1862/T1862_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22439/M22439_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21349/M21349_12_E1_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/20/20_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/71/71_12_E3_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/71/71 _12_Microbiome_Survey.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/363/363 _12_Microbiome_Survey.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/434/434_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M24341/M24341_12_E2_Notes .pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/79/79_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/468/468_12_E2_Notes_Sheet.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/393/393_12_E1_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/464/464_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/168/168_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21780/M21780_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/47/47_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/N410/N410_12_E1_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22911/M22911_12_E3_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22911/M22911_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/452/452_12_E3_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/452/452_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21215/M21215_12_E2_Notes_Sheet.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23697/M23697_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/219/219_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/144/144_12_Microbiome_Survey.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/144/144_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M23387/M23387_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/591/591_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M21706/M21706_12_E2_Notes_Sheet.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/249/249_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/174/174_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/170/170_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/8/8_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/M22400/M22400_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/237/237_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/18/18_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/279/279_12_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/12_Months/225/225_12_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21587/M21587_6_E1_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/439/439_6_Notes_E2.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M23468/M23468_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N208/N208_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/423/423_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/119/119_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/199/199_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/107/107_6_Notes_E2.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/516/516_6_Notes_E2.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22380/M22380_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/441/441_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/441/441_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21659/M21659_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/471/471_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/471/471_6_E1_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24793/M24793_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/534/534_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N801/N801_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21056/M21056_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/59/59_6_E2_Notes .docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/572/572_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/226/226_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N863/N863_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/268/268_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24432/M24432_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24957/M24957_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M23112/M23112_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21071/M21071_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/311/311_6_E1_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/311/311_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22289/M22289_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21620/M21620_6_E1_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21603/M21603_E1_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/6/6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N850/N850_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21642/M21642_6_E2 Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21662/M21662_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22000/M22000_6_E2 Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N186/N186_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/547/547_6_E2 Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/49/49_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/19/19_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22272/M22272_6_E2 Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/235/235_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/17/17_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/378/378_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24926/M24926_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24283/M24283_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/140/140_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/624/624_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/1912/1912_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21752/M21752_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/354/354_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/5/5_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M25139/M25139_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/109/109_E3_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/109/109_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M25139-2/M25139-2_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21614/M21614_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24749/M24749_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/127/127_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/55/55_ 6 Month E2 Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/111/111_6_E2_Notes .docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21269/M21269_ 6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/153/153_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21663/M21663_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21663/M21663_6_ E3_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/562/562_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21146/M21146_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22179/M22179_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N643/N643_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/74/74_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22901/M22901_6_E1_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/601/601_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22624/M22624_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22822/M22822_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/335/335_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/406/406_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/373/373_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N564/N564_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N564/N564_6_E3_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/45/45_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/424/424_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N419/N419_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/638/638_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/196/196_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22383/M22383_6_E1_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/100/100_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21226/M21226_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21052/M21052_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/201/201_E3_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/201/201_E2_Notes .docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/614/614_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/336/336_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N194/N194_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21994/M21994_6_E2 Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21570/M21570_6_E1_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24718/M24718_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21152/M21152_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/350/350_6_E2 Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/421/421_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22440/M22440_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/218/218_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24542/M24542_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22381/M22381_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21228/M21228_6_E2_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21588/M21588_6_E2_notes .docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21660/M21660_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22439/M22439_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21349/M21349_6_E2_notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/20/20_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/71/71_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/363/363_6_E3_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/363/363_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M24341/M24341_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/426/426_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/426/426_6_E3_Notes_2.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/468/468_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/464/464_6_Notes_E2.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/168/168_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22382/M22382_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M21780/M21780_6_E2 Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/452/452_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/452/452_6_E1_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N231/N231_6_E3_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N231/N231_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/102/102_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/144/144_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/299/299_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M23387/M23387_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/591/591_6_E1_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/591/591_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/249/249_6_Notes_E1.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/249/249_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/174/174_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/170/170_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22400/M22400_6_E2 Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N367/N367_6_E2_Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/237/237_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22306/M22306_6 Month E2 Notes.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/18/18_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/M22588/M22588_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/225/225_6_E2_Notes.docx.pdf
/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential/6_Months/N853/N853_6_E2_Notes.docx.pdf
(base) [sd6701@bigpurple-ln2 ~]$ 
#python launch.py -n 1 -s 1 --launcher ssh -H hosts --sync-dst-dir Difacto111 build/difacto local.conf
#export PS_VERBOSE=2
python launch.py -n 5 -s 4 --launcher ssh -H ip_list.txt --sync-dst-dir Difacto666 build/difacto local.conf 

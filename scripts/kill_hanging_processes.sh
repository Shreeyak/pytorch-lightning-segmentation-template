# When using DDP, processes don't always die cleanly. This snippet will kill all the processes of seg_lapa package.
kill $(ps aux | grep seg_lapa | grep -v grep | awk '{print $2}')

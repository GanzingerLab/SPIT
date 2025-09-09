import glob
import os
from spit import tools
from natsort import natsorted


def createTable(path):
    print('Creating Table for Wiki')
    filelist_or = natsorted(glob.glob(path + '/**/*result.txt', recursive=True))
    filelist  = []
    for i in filelist_or:
        if 'output' not in i:
            filelist.append(i)
    print('There are '+str(len(filelist))+' Run folders with a result.txt file.')

    table_header = '{| class="wikitable sortable mw-collapsible mw-collapsed"  \n |- \n \
        | bgcolor="#9fddf4" | Filepath || bgcolor="#9fddf4" colspan="10" | '+path+'</span>  \n |- \n \
            !Run Number!!Sample!!Mode!!Exposure!!Bitdepth!!Frames!!Lasers!!Frap!!FRAP Laser!!FRAP Power!!FRAP Duration\n |- \n'

    # create text file to store output in
    text_file = open(os.path.join(path, 'wiki_table.txt'), 'w', encoding='utf-8')
    text_file.write(table_header)

    for i in range(0, len(filelist)):  # loop through all result.txt files
        mode = '-'
        frap = '-'
        comment = '-'
        int_path = filelist[i].replace(path, '')[:-29]#r'\\'.join(.split(r'\\')[:-2])
        if 'output' in filelist[i]:
            continue
        with open(filelist[i]) as file:  # read in result.txt file and store in list
            analysisplot0 = '-'
            analysisplot1 = '-'
            analysisplot2 = '-'
            data_lines = file.readlines()
            run_number = filelist[i][-19:-11]
            sample = tools.find_string(data_lines, 'Sample:')[17:-1]
            mode = tools.find_string(data_lines, 'Mode')[17:20]
            exposure = tools.find_string(data_lines, 'Camera Exposure')[17:-1]
            ROI = tools.find_string(data_lines, 'Readout')[17:-1]
            comment = tools.find_string(data_lines, 'Comment')[17:-1]
            frap = tools.find_string(data_lines, 'FRAP:')[17:-1]
            time = tools.find_string(data_lines, 'DateTime:')[-9:-1]
            try:
                abortion = tools.find_string(data_lines, 'Aborted')[:-1]
            except:
                abortion = ''
            try:
                conclusion = tools.find_string(data_lines, 'Conclusion:')[17:-1]
            except:
                conclusion = '-'
            if frap == 'YES':
                frap_bold = '<b>YES</b>'
                analysisplot0 = '[[File:' + \
                    os.path.split(path)[1]+'_'+run_number+'FRAP.png|90px]] '
            else:
                frap_bold = 'NO'
            try:  # line added later on (after 2020-09-22)
                bitdepth = tools.find_string(data_lines, 'BitDepth')[24:-1]
            except:
                bitdepth = 'n.a.'
            try:  # FRAP measurement will produce these variables, otherwise they don't exist
                frap_laser = tools.find_string(data_lines, 'Laser(s):')[17:-1]
                frap_power = tools.find_string(data_lines, 'Power(s):')[17:-1]
                frap_duration = tools.find_string(data_lines, 'Duration')[17:-1]
            except:
                frap_laser = '-'
                frap_power = '-'
                frap_duration = '-'
            if mode == 'VCR':  # laser layout in txt file for VCR mode  20210928 Jurkat 06_Run00010_record_ch0_localization.png
                analysisplot0 = '[[File:'+os.path.split(
                    path)[1]+'_'+run_number+'_record_ch0_localization.png|90px]]  '
                analysisplot1 = '[[File:'+os.path.split(
                    path)[1]+'_'+run_number+'_record_ch1_localization.png|90px]]  '
                analysisplot2 = '[[File:'+os.path.split(
                    path)[1]+'_'+run_number+'_record_ch2_localization.png|90px]] '
                mode_color = '#800000'
                frames = tools.find_string(data_lines, 'Record Length')[17:-1]
                laser405 = tools.find_string(data_lines, 'Laser 405nm:     O')[17:-1]
                laser488 = tools.find_string(data_lines, 'Laser 488nm:     O')[17:-1]
                laser638 = tools.find_string(data_lines, 'Laser 638nm:     O')[17:-1]
                if tools.find_string(data_lines, 'Laser 561nm:     O')[17:20] == 'OFF' or tools.find_string(data_lines, 'Laser 561nm:     O')[17:20] == 'Off':
                    laser561 = 'OFF'
                else:
                    laser561 = tools.find_string(
                        data_lines, 'Laser 561nm:     Emitting at')[29:-1]
                lasers = '405nm:' + laser405+', 488nm: '+laser488+', 561nm: '+laser561+', 638nm:'+laser638
            if mode == 'Seq':  # laser and timing layout in txt file for Sequential mode
                mode_color = '#008080'
                interval = tools.find_string(data_lines, 'Interval')[17:-1]
                mode = mode+'<br> interval '+interval
                more_patterns = ''
                frames = tools.find_string(data_lines, 'Acquisitions')[17:-1]
                laser405 = tools.find_string(data_lines, 'Laser 405nm')[40:-1]
                laser488 = tools.find_string(data_lines, 'Laser 488nm')[40:-1]
                laser638 = tools.find_string(data_lines, 'Laser 638nm')[40:-1]
                laser561 = tools.find_string(data_lines, 'Laser 561nm')[29:-1]
                pattern = 'P1: '+tools.find_string(data_lines, 'Pattern01')[17:-1]
                try:  # if there is more then one step in the pattern
                    pattern = pattern+', P2: ' + \
                        tools.find_string(data_lines, 'Pattern02')[17:-1]
                    pattern = pattern+', P3: ' + \
                        tools.find_string(data_lines, 'Pattern03')[17:-1]
                except:
                    pattern = pattern
                lasers = pattern+' <br> 405: '+laser405+', 488: ' + \
                    laser488+', 561: '+laser561+', 638: '+laser638
            if mode == 'zSt':  # zStack
                mode_color = '#808000'
                frames = tools.find_string(data_lines, 'Steps')[17:-1]+' steps'
                laser405 = tools.find_string(data_lines, 'Laser 405nm:     O')[17:-1]
                laser488 = tools.find_string(data_lines, 'Laser 488nm:     O')[17:-1]
                laser638 = tools.find_string(data_lines, 'Laser 638nm:     O')[17:-1]
                if tools.find_string(data_lines, 'Laser 561nm:     O')[17:20] == 'OFF' or tools.find_string(data_lines, 'Laser 561nm:     O')[17:20] == 'Off':
                    laser561 = 'OFF'
                else:
                    laser561 = tools.find_string(
                        data_lines, 'Laser 561nm:     Emitting at')[29:-1]
                lasers = '405nm: '+laser405+', 488nm: '+laser488+', 561nm: '+laser561+', 638nm:'+laser638

            mod = i % 2
            background_color = ['#d8d8b2', '#d8c5b2']

        text_file.write('| bgcolor="'+background_color[mod]+'" rowspan="5" | '+run_number+' <br> {{small|' + time + '}} || bgcolor="'+background_color[mod]+'" |'+sample+'|| bgcolor="'+background_color[mod]+'" |'+mode+'||bgcolor="'+background_color[mod]+'" |'+exposure+'||bgcolor="'+background_color[mod]+'" |'+bitdepth+'||bgcolor="'+background_color[mod]+'" |'+frames+'|| bgcolor="'+background_color[mod]+'" |'
                        + lasers+'|| bgcolor="'+background_color[mod]+'" |' + frap_bold+' ||bgcolor="'+background_color[mod]+'" |'+frap_laser+'||bgcolor="'+background_color[mod]+'" |'+frap_power+'||bgcolor="'+background_color[mod]+'" |'+frap_duration+ ' \n |- \n')
        text_file.write(
            '| bgcolor="'+background_color[mod]+'" colspan="10" | Comment:  <span style="color:#0a78a0">'+comment+'. '+abortion+'</span> \n |-\n')
        text_file.write(
            '| bgcolor="'+background_color[mod]+'" colspan="10" | Conclusion:  <span style="color:#0a78a0">'+conclusion+'.</span> \n |-\n')
        text_file.write(
            '| bgcolor="'+background_color[mod]+'" colspan="10" | path:  <span style="color:#0a78a0">'+int_path+'</span> \n |-\n')
        text_file.write(
            '| bgcolor="'+background_color[mod]+'" colspan="10" | Manual_comment:  <span style="color:#0a78a0">'+''+'</span> \n |-\n')
    text_file.write('|}')

    text_file.close()
    print('Created wiki_table.txt file')

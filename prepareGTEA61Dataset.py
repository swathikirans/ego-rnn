import os
import glob

label_dir = './labels' # Dir containing the labels
src_img_dir = './images/frames' # Dir containing the frames/flow
des_img_dir = './gtea_dataset/frames' # Dir to which the images have to be copied

label_files = glob.glob1(label_dir, '*.txt')
action_inst = {}
for label_file in sorted(label_files):
    print(label_file)
    file = os.path.join(label_dir, label_file)
    sub_dir = label_file[:2]
    label_fl = open(file, 'r')
    annots = label_fl.readlines()
    for line1 in annots:
        if '><' in line1:
            w1b = line1.find('<')
            w2b = line1.find('<', w1b+1)
            w1e = line1.find('>')
            w2e = line1.find('>', w1e+1)
            action_name = line1[w1b+1:w1e] + '_' + line1[w2b+1:w2e]
            if action_name == 'stir_cup':
                action_name = 'stir_spoon,cup'
            action_dir = os.path.join(des_img_dir, sub_dir, action_name)
            if not os.path.exists(action_dir):
                os.makedirs(action_dir)
                action_inst[action_name] = 1
            des_dir = os.path.join(action_dir, str(action_inst[action_name]))
            action_inst[action_name] += 1
            os.makedirs(des_dir)
            f1 = line1.find('(')
            f2 = line1.find('-')
            f3 = line1.find(')')
            start_frame = line1[f1+1:f2]
            end_frame = line1[f2+1:f3]
            frame_ind = 0
            for f in range(int(start_frame), int(end_frame)+1):
                frame_name = src_img_dir + '/' + label_file[:-4] + '/' + str(f).zfill(5) + '.jpg'
                cmd = 'cp ' + frame_name + ' ' + des_dir + '/image_' + str(frame_ind).zfill(5) + '.jpg'
                os.system(cmd)
                frame_ind += 1

os.system('rm -rf ' + des_img_dir + '/S1/put_tea')
os.system('rm -rf ' + des_img_dir + '/S2/put_tea')

import os
import random
import argparse
'''


'''
def adjust_label(lines):
    for index,line in enumerate(lines):
        line = line.rstrip();
        if line=='':
            continue
            print "this line is empty"
        words = line.split('\t');
        print words
        print words[1]
        words[1] = int(words[1])-1;
        lines[index] = '%s\t%d\n'%(words[0], words[1]);
    return lines;

def split(input_url,output_dir,split_size=128,adjustLabel=False):

    f = open(input_url,'r');
    lines = f.readlines()
    f.close()

    random.shuffle(lines)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir);

    count = 0
    batch_count=0
    while count < len(lines):
        if count + split_size - 1 < len(lines):
            new_lines = lines[count:count + split_size];
            if adjustLabel:
                new_lines = adjust_label(new_lines);
            new_lines[len(new_lines) - 1] = new_lines[len(new_lines) - 1].rstrip();
            output_path = '%s/split1_test_%d.txt' % (output_dir, batch_count);
            new_f = open(output_path, 'w');
            new_f.writelines(new_lines);
            batch_count+=1
        else:
            new_lines = lines[count::];
            if adjustLabel:
                new_lines = adjust_label(new_lines);
            new_lines[len(new_lines) - 1] = new_lines[len(new_lines) - 1].rstrip();
            output_path = '%s/split1_test_%d.txt' % (output_dir, batch_count);
            new_f = open(output_path, 'w');
            new_f.writelines(new_lines);
        count+=split_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_url", required=True, help='each labeling project has one uniqe project_id')
    parser.add_argument("--output_dir", required=True, help='video list fetched from mysql or cluster')
    args = parser.parse_args()
    input_url = args.input_url
    output_dir = args.output_dir
    split_size = 512
    adjustLabel = False
    split(input_url,output_dir,split_size)

    # else:
    #     new_lines  = lines[count:count+batch_num-1];
    #     if adjustLabel:
    #         new_lines = adjust_label(new_lines);
    #     new_lines[len(new_lines) - 1] = new_lines[len(new_lines) - 1].rstrip();
    #     if len(new_lines)>0:
    #         output_path = '%s/split_%d.txt' % (output_dir, batch_count);
    #         new_f = open(output_path, 'w');
    #         new_f.writelines(new_lines);
    #         new_f.close();
    #         count += batch_num;
    #         batch_count += 1;


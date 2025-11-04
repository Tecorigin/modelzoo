import os
import hashlib
# from prototxt_parser.prototxt_parser_main import parse
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--cases_dir', dest='cases_dir', type=str, help='cases dir')
args = parser.parse_args()

def read_from_txt(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines
    except Exception as e:
        print(e)
        return []

def read_from_dir(dirs):
    try :
        cmd = f"find {dirs} -name '*.prototxt' -type f"
        read_process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if read_process.returncode == 0:
            return read_process.stdout.split()
        else:
            print("no prototxt found")
            return []
    except Exception as e:
        print(e)
        return []

def read(filename):
    content = ''
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "upper_bound" in line:
                pos = line.find("upper_bound")
                line = " " * pos + "upper_bound: 10\n"
            if "lower_bound" in line:
                pos = line.find("lower_bound")
                line = " " * pos + "lower_bound: -10\n"
            content += line
    return content

def get_hash(string):
    m = hashlib.md5()
    m.update(string.encode('utf-8'))
    return m.hexdigest()

def dumplicate_op(input_path, output_path):
    if not os.path.exists(input_path):
        print("input_path don't exist: {input_path}")
        return
    test_case = read_from_dir(input_path)
    test_case.sort()

    cases_num_before = len(test_case)
    if cases_num_before == 0:
        print("input_path has no cases: {input_path}")
        return

    cases_hash = []
    all_cases = []
    cases_hash_dict = dict()
    all_cases_dict = dict()
    num=0;
    for case_path in test_case:
        # print(case_path)
        content = read(case_path)
        case_hash = get_hash(content)
        num=num+1;
        if case_hash not in cases_hash:
            cases_hash.append(case_hash)
            all_cases.append(case_path)
            cases_hash_dict[case_hash] = case_path
            all_cases_dict[case_path] = 1
        else:
            all_cases_dict[cases_hash_dict[case_hash]] += 1

    cases_num_after = len(all_cases)
    print(f"cases_before_num:{cases_num_before}, cases_after_num:{cases_num_after}")

    save_name = os.path.join(output_path, "cases_list.txt")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(save_name, 'w') as f:
        data = ""
        for case in all_cases:
            case_name = case[case.rfind('/')+1:]
            op_name = case[:case.rfind('/')]
            op_name = op_name[op_name.rfind("/")+1:]
            testcase_num_str = "{:80} {} \n".format(case_name, all_cases_dict[case])
            data += testcase_num_str
            if True:
                os.system(f"cp {case} {output_path}")
        f.write(data)
        print("{} {} case num saved in {}!".format(op_name, cases_num_after, save_name))

def dumplicate_model(model_path, input_path = "dump_info", output_path = "dump_info_dedumplicate"):
    input_fullpath = os.path.join(model_path, input_path)
    output_fullpath = os.path.join(model_path, output_path)

    if not os.path.exists(input_fullpath):
        print("input_path don't exist: {input_fullpath}")
        return
    if not os.path.exists(output_fullpath):
        os.mkdir(output_fullpath)

    ops = list()
    files = os.listdir(input_fullpath)
    for file in files:
        file_path = os.path.join(input_fullpath, file)

        if os.path.isdir(file_path):
            ops.append(file)

    ops.sort()
    for op in ops:
        op_input_path = os.path.join(input_fullpath, op)
        op_output_path = os.path.join(output_fullpath, op)

        dumplicate_op(op_input_path, op_output_path)

if __name__ == "__main__":
    if args.cases_dir != "":
        dumplicate_model(args.cases_dir)
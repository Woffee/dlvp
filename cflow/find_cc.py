"""
基于 jy 的代码，确保 lv0 开始的数据就是一致的。

Dependencies:
    GitPython >= 3.1.18
    smmap >= 4.0.0
    gitdb >= 4.0.7
    whatthepatch >= 1.0.2


@Time    : 7/15/21
@Author  : Wenbo
"""


# environment setup
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import *
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(BASE_DIR, "functions_wenbo")
Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)



# project configure
taskDescriptor: str = os.path.join(BASE_DIR, "tasks.json")
"""
    Path to task description list.
"""


repoDirectory: str = os.path.join(BASE_DIR, "repo")
"""
    Path to repo store directory.
"""


versDirectory: str = os.path.join(BASE_DIR, "verstehen")
"""
    Path to verstehen report store directory.
"""


# undbDirectory: str = "/data/jiyuan/undb"
"""
    Path to Understand DB store directory.
"""


depthLevel: int = 3
"""
    Depth of caller/callee tree.
"""


# global configure
sitepkgPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'site-packages')
"""
    Path to the installed wheels.
    As Understand Python does not supports pip, you need to specify the search path here.
    Packages should already bundled with the script, so no modification is needed.
"""


# imports
sys.path.append(sitepkgPath)
import git as git
import whatthepatch as patch


# aliases
DiffObj = patch.patch.diffobj


@dataclass
class Function:
    startLine: int
    endLine: int
    name: str
    uniquename: str
    contents: str
    type: str
    parameters: str
    kind: str
    commit: str

class MyFunction:
    def __init__(self, func_name, file_name="", file_loc=0, code=""):
        self.func_name = func_name
        self.file_name = file_name
        self.file_loc = file_loc
        self.code = code
        self.callees = []
        self.callers = []

# common infra
# def dbnormpath(file: und.Ent, base: str) -> str:
#     """
#     Normalize Ent paths
#     """
#     return os.path.normpath(os.path.relpath(file.longname(), base))


def gitnormpath(file: str) -> str:
    """
    Normalize Git paths
    """
    return os.path.normpath(file)


def createdb(project: str, output: str, lang: str) -> None:
    """
    Create an Understand database for the project
    """
    os.system(f"und create -languages {lang} \"{output}\"")
    os.system(f"und add \"{project}\" \"{output}\"")
    os.system(f"und analyze \"{output}\"")


def updatedb(project: str, output: str) -> None:
    """
    Incrementally update the Understand database
    """
    os.system(f"und analyze -changed \"{output}\"")

# private utility
def diff_transformer(diff: DiffObj) -> List[Tuple[int, int]]:
    """
    Create line-of-interests from a DiffObj
    """
    result = []
    last_new = -1
    is_del_section = False

    for c in diff.changes:
        if c.new is None:
            is_del_section = True
        else:
            if is_del_section:
                is_del_section = False
                result += [(last_new, c.new)]
            last_new = c.new

        if c.old is None:
            result += [(c.new, c.new)]
    return result


def unique(list1: List) -> List:
    """
    List unique items of a list
    """
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


# def extract_range(func: und.Ent) -> Tuple[int, str, int]:
#     start_rel = func.refs("definein")
#     start_rel2 = func.refs("begin,definein")
#
#     start_ln = -1
#     if len(start_rel) > 0:
#         start_ln = start_rel[0].line()
#
#     contents_fn = func.contents()
#     end_ln = start_ln + contents_fn.count('\n')
#
#     return start_ln, contents_fn, end_ln


# def function_transform(func: und.Ent, commit: str) -> Function:
#     """
#     Create a JSON-friendly function entity object
#     """
#     start_ln, contents_fn, end_ln = extract_range(func)
#
#     entity = Function(
#         start_ln,
#         end_ln,
#         func.name(),
#         func.uniquename(),
#         contents_fn,
#         func.type(),
#         func.parameters(),
#         str(func.kind()),
#         commit
#     )
#     return asdict(entity)


def rel(type: str, value: str) -> Dict[str, str]:
    """
    Create a JSON-friendly relation object
    """
    return {"type": type, "value": value}


# git abstract layer
class GitRepo:
    """
    Represents a Git repository

    Usage:
        repo = GitRepo(projectPath)
        cur = repo.head()
        cur = repo.checkout(commitId)
        prev = repo.prev(cur)
        diff = repo.diff(prev, cur)
        files = repo.changed_files(diff)
        detail = repo.changed_lines(diff[0])
    """

    def __init__(self, path: str):
        self.repo = git.Repo(path)

    def head(self) -> git.Commit:
        """
        Get current HEAD
        Equivalent:
            `git rev-parse HEAD`
        """
        return self.repo.head.commit

    def checkout(self, commit: git.Commit) -> git.Commit:
        """
        Checkout a commit and set HEAD to it
        Also:
            `checkout(self, commit: str) -> git.Commit`
        Equivalent:
            `git checkout commit`
        """
        self.repo.git.checkout(commit, force=True)
        return self.head()

    def prev(self, commit: git.Commit) -> git.Commit:
        """
        Get parent commit of a specific commit
        Equivalent:
            `git rev-parse commit~1`
        """
        # todo: multi parents?
        parents = commit.parents
        if len(parents) <= 0:
            return None
        else:
            return parents[0]

    def diff(self, older: git.Commit, newer: git.Commit) -> git.DiffIndex:
        """
        Diff two commits
        Equivalent:
            `git diff older newer`
        """
        return older.diff(newer, create_patch=True)

    def changed_files(self, diff: git.DiffIndex) -> List[str]:
        """
        Get changed files of a diff index
        Equivalent:
            None
        """
        paths = []
        # todo: deleted, copied
        for d in diff.iter_change_type("A"): # added
            paths += [gitnormpath(d.b_path)]
        for d in diff.iter_change_type("R"): # renamed
            paths += [gitnormpath(d.b_path)]
        for d in diff.iter_change_type("M"): # modified
            paths += [gitnormpath(d.b_path)]
        return paths

    def changed_file_lines(self, diff: git.Diff) -> List[DiffObj]:
        """
        Get changed lines of a single diff
        Equivalent:
            None
        """
        try:
            parseGen = patch.parse_patch(diff.diff.decode("utf-8"))
            parseLst = []
            for p in parseGen:
                parseLst += diff_transformer(p)
            return parseLst
        except:
            return []

    def changed_lines(self, diff: git.DiffIndex) -> Dict[str, DiffObj]:
        """
        Get changed lines of a diff index
        Equivalent:
            None
        """
        changes = {}
        for d in diff.iter_change_type("A"): # added
            changes[gitnormpath(d.b_path)] = self.changed_file_lines(d)
        for d in diff.iter_change_type("R"): # renamed
            changes[gitnormpath(d.b_path)] = self.changed_file_lines(d)
        for d in diff.iter_change_type("M"): # modified
            changes[gitnormpath(d.b_path)] = self.changed_file_lines(d)
        return changes


# job runner
class JobRunner:
    """
    Run analyze job on a given Understand database

    Description:
        For each modified function, recursively list their implementations and call-relations

    Usage:
        cur = repo.checkout(commit)
        prev = repo.prev(cur)
        diff = repo.diff(prev, cur)
        runner.commit_job(cur.hexsha, diff)
    """

    def __init__(self, repo: git.Repo, proj: str, undb: str, depth: int):
        self.proj = proj
        # self.undb = undb
        self.repo = repo
        self.depth = depth
        self.uobj = None
        self.commitId = ""
        self.entReg = {}
        self.relReg = {}

    # def id(self, func: und.Ent) -> str:
    #     """
    #     Get commit-aware object id
    #     """
    #     return self.commitId + func.uniquename()

    # def _func_job(self, func: und.Ent) -> None:
    #     """
    #     Function job.
    #
    #     Description:
    #         Recursively list their implementations and call-relations for a function.
    #     """
    #     call = [func]
    #     callby = [func]
    #     self.entReg[self.id(func)] = function_transform(func, self.commitId)
    #
    #     for ii in range(0, self.depth):
    #         dummy_call = []
    #         for u in call:
    #             call_id = unique([i.ent().id() for i in u.refs("call", "function,method,procedure")])
    #             call_ent = [self.uobj.ent_from_id(i) for i in call_id]
    #
    #             for c in call_ent:
    #                 self.entReg[self.id(c)] = function_transform(c, self.commitId)
    #
    #                 if self.id(u) not in self.relReg:
    #                     self.relReg[self.id(u)] = []
    #
    #                 self.relReg[self.id(u)] += [rel("call", self.id(c))]
    #                 dummy_call += [c]
    #
    #         dummy_callby = []
    #         for u in callby:
    #             callby_id = unique([i.ent().id() for i in u.refs("callby", "function,method,procedure")])
    #             callby_ent = [self.uobj.ent_from_id(i) for i in callby_id]
    #             for c in callby_ent:
    #                 self.entReg[self.id(c)] = function_transform(c, self.commitId)
    #
    #                 if self.id(u) not in self.relReg:
    #                     self.relReg[self.id(u)] = []
    #
    #                 self.relReg[self.id(u)] += [rel("callby", self.id(c))]
    #                 dummy_callby += [c]
    #         call = dummy_call
    #         callby = dummy_callby
    #
    #
    # def _file_job(self, file: und.Ent, beacons: List[Tuple[int, int]]) -> None:
    #     """
    #     File job.
    #
    #     Description:
    #         Find modified functions in a file and execute function jobs on them.
    #     """
    #     funcs = file.ents("define", "function,method,procedure")
    #     self.relReg[self.id(file)] = []
    #
    #     for f in funcs:
    #         start_ln, contents_fn, end_ln = extract_range(f)
    #
    #         modified = False
    #         for i in beacons:
    #             if start_ln <= i[0] and i[1] <= end_ln:
    #                 modified = True
    #                 break
    #
    #         if not modified:
    #             continue
    #
    #         self.relReg[self.id(file)] += [rel("define", self.id(f))]
    #         self._func_job(f)

    def commit_job(self, cmtId: str, beacons: git.DiffIndex) -> None:
        """
        Commit job.

        Description:
            Find modified files in a commit and execute file jobs on them.
        """
        self.commitId = cmtId
        # updatedb(self.proj, self.undb)

        files = self.repo.changed_files(beacons)
        lines = self.repo.changed_lines(beacons)

        # db = und.open(self.undb)
        # self.uobj = db

        self.relReg[self.commitId] = []

        for file in files:
            print("file: ",file)
            print("lines:", lines)

            # candidates = db.lookup(os.path.basename(file), "file")
            # candidate = None
            #
            # for c in candidates:
            #     if dbnormpath(c, self.proj) == file:
            #         candidate = c
            # if candidate is None:
            #     continue

            # self.relReg[self.commitId] += [rel("change", self.id(candidate))]
            # self._file_job(candidate, lines[file])




"""
# code to fix legacy line ending problem

def line_end_fix(outputDir: str):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(outputDir):
        f.extend(filenames)

    r = []
    for filename in f:
        filename = str(filename)
        if filename.endswith('entities.json'):
            r += [os.path.join(outputDir, filename)]

    for filename in r:
        with open(filename, 'r+') as entFile:
            entReg = json.load(entFile)
            for k in entReg:
                v = entReg[k]
                start_ln = v['startLine']
                content = v['contents']
                v['endLine'] = start_ln + str(content).count('\n')
                entReg[k] = v
            entFile.truncate(0)
            entFile.seek(0)
            json.dump(entReg, entFile)

line_end_fix()
"""

def tree2relations(entities, relations, funcs, vul, lv=0):
    """
    :param entities:
    :param relations: e.g. {
                              "7ba100d3e6e8b1e5d5342feb960a7f081d6e15af": [
                                {
                                  "type": "define", # level 0
                                  "value": "7ba100d3e6e8b1e5d5342feb960a7f081d6e15af@l./../repo/FFmpeg/libavformat/hls.c"
                                }
                              ],
                              "7ba100d3e6e8b1e5d5342feb960a7f081d6e15af@lread_data@p@l./../repo/FFmpeg/libavformat/hls.c": [
                                {
                                  "type": "call",   # or "callby"
                                  "value": "7ba100d3e6e8b1e5d5342feb960a7f081d6e15af@lav_log@kav_log@f./../repo/FFmpeg/libavutil/log.c"
                                },
                                ...
                               ]
                            }
    :param funcs: each function:
                        {
                            "commit_id": commit,
                            "func_name": func.func_name,
                            "file_name": func.file_name,
                            "file_loc": func.file_loc,

                            "code": func.code,
                            "callers": [],
                            "callees": []
                        }
    :param vul: 是否是 vulnerable function
    :param lv: level

    :return:
    """

    """ 
    """
    for func in funcs:
        uniquename = func['commit_id'] + "@" + func['file_name'] + "@" + func['func_name']

        # save to entities:
        if uniquename not in entities.keys():
            entities[uniquename] = {
                "name": func['func_name'],
                "uniquename": uniquename,
                "commit": func['commit_id'],
                "vul": vul,

                "file_name": func['file_name'],
                "file_loc": func['file_loc'],
                "contents": func['code'],
            }

        # save to relations:
        if func['commit_id'] not in relations.keys():
            relations[func['commit_id']] = []
        if uniquename not in relations.keys():
            relations[uniquename] = []
        if lv == 0:
            relations[func['commit_id']].append({
                "type": "define",
                "value": uniquename
            })

        if len(func['callees']) > 0:
            for sub_func in func['callees']:
                sub_uniquename = sub_func['commit_id'] + "@" + sub_func['file_name'] + "@" + sub_func['func_name']
                relations[uniquename].append({
                    "type": "call",
                    "value": sub_uniquename
                })
            entities, relations = tree2relations(entities, relations, func['callees'], vul, lv + 1)

        if len(func['callers']) > 0:
            for sub_func in func['callers']:
                sub_uniquename = sub_func['commit_id'] + "@" + sub_func['file_name'] + "@" + sub_func['func_name']
                relations[uniquename].append({
                    "type": "callby",
                    "value": sub_uniquename
                })
            entities, relations = tree2relations(entities, relations, func['callers'], vul, lv + 1)
    return entities, relations


def get_space_num(line):
    n = 0
    for c in line:
        if c == " ":
            n += 1
        else:
            break
    return n


def parse_cflow_line(line):
    p1 = re.compile(r'.*?[(][)]', re.S)  # func name
    p2 = re.compile(r'at .*?[.]c:.*?>', re.S)  # filename and locations

    func_name = ""
    file_name = ""
    file_loc = "0"

    res = re.findall(p1, line)
    if len(res) > 0:
        func_name = res[0].strip()

        res = re.findall(p2, line)
        if len(res) > 0:
            file_name_loc = res[0].strip()[3:-1]
            # print(file_name_loc)
            file_name, file_loc = file_name_loc.split(":")
            # print(file_name, file_loc)

    return func_name, file_name, file_loc

def _recurse_tree(parent, depth, source, c_type=0):
    last_line = source.readline().replace("\t", "    ").rstrip()
    while last_line:
        tabs = get_space_num(last_line)
        if tabs < depth:
            break
        # node = last_line.strip()
        func_name, file_name, file_loc = parse_cflow_line(last_line.strip())
        # code 先设置为空，后面再补充
        code = ""
        sub_func = MyFunction(func_name, file_name, file_loc, code)

        if tabs >= depth:
            if parent is not None:
                # print("%s: %s" %(parent, node))
                if c_type == 0:
                    parent.callees.append(sub_func)
                else:
                    parent.callers.append(sub_func)
            last_line = _recurse_tree(sub_func, tabs + 1, source, c_type)
    return last_line


# read a function code from a c file.
# https://stackoverflow.com/questions/55078713/extract-function-code-from-c-sourcecode-file-with-python
def process_file(filename, line_num):
    if not os.path.exists(filename):
        return ""

    print("opening " + filename + " on line " + str(line_num))

    code = ""
    cnt_braket = 0
    found_start = False
    found_end = False

    # encoding = "ISO-8859-1" for xye server
    with open(filename, "r", encoding="utf8", errors='ignore') as f:
        for i, line in enumerate(f):
            if (i >= (line_num - 1)):
                code += line

                if line.count("{") > 0:
                    found_start = True
                    cnt_braket += line.count("{")

                if line.count("}") > 0:
                    cnt_braket -= line.count("}")

                if cnt_braket == 0 and found_start == True:
                    found_end = True
                    print("== len of code: %d" % len(code))
                    return code
    print("== len of code: %d" % len(code))
    return code


def recurse_to_dict(project_path, commit, func, changed_func_names, depth):
    dd = {
        "commit_id": commit,
        "func_name": func.func_name,
        "file_name": func.file_name,
        "file_loc": func.file_loc,

        "code": func.code,
        "callers": [],
        "callees": []
    }
    if len(func.callees) > 0:
        for ff in func.callees:
            # print("==", depth, ff.file_name, changed_func_names)
            if depth == 0 and changed_func_names is not None and ff.file_name.strip() != "" and ff.file_name in changed_func_names.keys() and ff.func_name in \
                    changed_func_names[ff.file_name]:
                ff.code = process_file(project_path + ff.file_name[1:], int(ff.file_loc))
                dd['callees'].append(recurse_to_dict(project_path, commit, ff, changed_func_names, depth + 1))
            if depth > 0 and ff.file_name.strip() != "":
                ff.code = process_file(project_path + ff.file_name[1:], int(ff.file_loc))
                dd['callees'].append(recurse_to_dict(project_path, commit, ff, changed_func_names, depth + 1))

    if len(func.callers) > 0:
        for ff in func.callers:
            # print("==", ff.file_name)
            if depth == 0 and changed_func_names is not None and ff.file_name.strip() != "" and ff.file_name in changed_func_names.keys() and ff.func_name in \
                    changed_func_names[ff.file_name]:
                ff.code = process_file(project_path + ff.file_name[1:], int(ff.file_loc))
                dd['callers'].append(recurse_to_dict(project_path, commit, ff, changed_func_names, depth + 1))
            if depth > 0 and ff.file_name.strip() != "":
                ff.code = process_file(project_path + ff.file_name[1:], int(ff.file_loc))
                dd['callers'].append(recurse_to_dict(project_path, commit, ff, changed_func_names, depth + 1))
    return dd

def find_caller_callee(commit, functions, data_type="ffmpeg", project_path=""):
    show_log_savepath = BASE_DIR + "/tmp"

    # logger.info("=== find_caller_callee")
    changed_funcs = {}
    p1 = re.compile(r'[^ ]*?[(]', re.S)
    for ff in functions:
        filename, b = ff.split(":::")
        res = re.findall(p1, b)
        if len(res) > 0:
            func_name = res[0] + ")"
            if filename not in changed_funcs.keys():
                changed_funcs[filename] = [func_name]
            else:
                changed_funcs[filename].append(func_name)
    # print("changed_funcs:")
    # print(changed_funcs)

    cmd = "cd %s && " % project_path + 'find . -name "*.c"'
    p = os.popen(cmd)
    x = p.read()

    c_files = x.strip().split("\n")

    # callees
    cmd = "cd %s && " % project_path + "cflow "
    paths = []
    for filename in c_files:
        pp = os.path.dirname(filename)
        if pp not in paths:
            paths.append(pp)
            # print(pp)
            cmd = cmd + pp + "/*.c "

    to_callee_file = "%s/%s_callee.txt" % (show_log_savepath, commit)
    cmd = cmd + " --omit-arguments --depth=3 --all --all"
    print(cmd)
    # logger.info("cmd: %s --output=%s" % (cmd, to_callee_file))
    p = os.popen(cmd + " --output=" + to_callee_file)
    x = p.read()

    print("saved to", to_callee_file)

    func_callee = MyFunction("root_callee")
    with open(to_callee_file) as inFile:
        _recurse_tree(func_callee, 0, inFile, 0)

    # callers
    to_caller_file = "%s/%s_caller.txt" % (show_log_savepath, commit)

    cmd += " --reverse "
    print(cmd)
    # logger.info("cmd: %s --output=%s" % (cmd, to_caller_file))
    p = os.popen(cmd + " --output=" + to_caller_file)
    x = p.read()

    print("saved to", to_caller_file)
    # logger.info("saved to %s" % to_caller_file)

    func_caller = MyFunction("root_caller")
    with open(to_caller_file) as inFile:
        _recurse_tree(func_caller, 0, inFile, 1)

    func_callee_dict = recurse_to_dict(project_path, commit, func_callee, changed_funcs, 0)
    func_caller_dict = recurse_to_dict(project_path, commit, func_caller, changed_funcs, 0)
    # logger.info("=== len of  func_callee_dict['callees']: %d" % len(func_callee_dict['callees']))
    # logger.info("=== len of  func_callee_dict['callers']: %d" % len(func_callee_dict['callers']))
    # logger.info("=== len of  func_caller_dict['callees']: %d" % len(func_caller_dict['callees']))
    # logger.info("=== len of  func_caller_dict['callers']: %d" % len(func_caller_dict['callers']))

    d1 = func_callee_dict['callees']
    d2 = func_caller_dict['callers']

    return d1 + d2

# batch job
def batch_job(tasks, project, underdb, output, lang) -> None:
    """
    Batch job.

    Description:
        Execute commit jobs, collect the result and write to files.
    """
    logging.info("project: %s" % project)
    if not os.path.exists(project):
        logging.error("project not existed: %s" % project)
        return

    repo = GitRepo(project)


    start_time = time.time()
    project_name = project.split("/")[-1]

    entities = {}
    relations = {}


    for task in tasks:

        taskName = task # CVE_ID
        taskCommits = tasks[task]

        save_path = os.path.join(SAVE_PATH, project_name)
        Path(save_path).mkdir(parents=True, exist_ok=True)
        to_entities_file = "%s/%s-entities.json" % (save_path, taskName)
        to_relation_file = "%s/%s-relation.json" % (save_path, taskName)

        print("Working on task: " + taskName)
        print("Project: " + project)
        logging.warning("Working on task: " + taskName)

        if os.path.exists(to_entities_file) and os.path.exists(to_relation_file):
            print("=== file existed: %s" % to_entities_file)
            logging.warning("=== file existed: %s" % to_entities_file)
            continue


        for commit in taskCommits:
            cur = repo.checkout(commit)
            prev = repo.prev(cur)
            print("cur:", cur)

            # diff = repo.diff(prev, cur)
            # print("diff:", diff)
            # runner.commit_job(cur.hexsha, diff)

            # git show diff
            cmd = "cd %s && git --no-pager show %s" % (project, commit)
            # cmd = " git --no-pager diff %s %s^1" % (commit, affected_tag)

            p = os.popen(cmd)
            try:
                x = p.read()
            except:
                x = ""
                logging.error("=== p.read() error, cmd: %s" % cmd)
            # print(x)


            # 【1】找出 修改的 filename 和 func name （计算 vul， non-vul distribution）
            changed_files_num = 0
            changed_func_num = 0
            changed_func_names = []
            is_c_file = False
            for line in x.strip().split("\n"):
                loc = line.find("+++ b/")
                if loc > -1:
                    filename = line[6:].strip()
                    not_test = True
                    if (filename.find("/tests/") > -1 or filename.find("/test/") > -1 or filename.find(
                            "/doc/") > -1):
                        not_test = False

                    if not_test and filename[-2:] == '.c':
                        changed_files_num += 1
                        is_c_file = True
                    else:
                        is_c_file = False

                if is_c_file and line.find("@@") > -1:
                    arr = line.strip().split("@@")
                    func_name = "./" + filename + ":::" + arr[-1].strip()
                    # print(arr[-1].strip())
                    if func_name not in changed_func_names:
                        changed_func_names.append(func_name)
                        print("changed func: %s" % func_name)
                        changed_func_num += 1

            functions_after = find_caller_callee(str(cur), changed_func_names, taskName, project)

            repo.checkout(prev)

            # diff = repo.diff(cur, prev)
            # runner.commit_job(prev.hexsha, diff)

            functions_before = find_caller_callee(str(prev), changed_func_names, taskName, project)

            entities, relations = tree2relations(entities, relations, functions_before, 1)
            entities, relations = tree2relations(entities, relations, functions_after, 0)

        # save to files
        with open(to_entities_file, 'w') as f:
            json.dump(entities, f)
        print("saved to", to_entities_file)
        logging.info("saved to: %s" % to_entities_file)

        with open(to_relation_file, 'w') as f:
            json.dump(relations, f)
        print("saved to", to_relation_file)
        logging.info("saved to: %s" % to_relation_file)


    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))


# task runner
logging.basicConfig(filename='trace.log', level=logging.DEBUG)
# Path(undbDirectory).mkdir(parents=True, exist_ok=True)

with open(taskDescriptor, 'r') as taskFile:
    taskDesc = json.load(taskFile)
    for repoName in taskDesc:
        # if repoName != 'FFmpeg':
        #     continue
        print("Working on repo: " + repoName)
        logging.warning("Working on repo: " + repoName)

        folder_name = taskDesc[repoName]['repo'].replace(".git", "").split("/")[-1]
        repoPath = os.path.join(repoDirectory, folder_name)
        # undbPath = os.path.join(undbDirectory, repoName + ".und")
        undbPath = ""
        versPath = os.path.join(versDirectory, repoName)
        Path(repoPath).mkdir(parents=True, exist_ok=True)
        # Path(undbPath).mkdir(parents=True, exist_ok=True) # do not create undb
        Path(versPath).mkdir(parents=True, exist_ok=True)

        batch_job(taskDesc[repoName]["vuln"], repoPath, undbPath, versPath, "C++")




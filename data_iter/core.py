#-*- coding: utf-8 -*-

import os
import json
import traceback

class ExceptionWrapper(object):
    "Wraps an exception plus traceback to communicate across threads"

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


class VideoSample(object):
    def __init__(self, video=None, video_url=None, cover_url=None, created_time=None, pl1_labels=None, pl2_labels=None, ml1_labels=None, ml2_labels=None, mvid=None):
        super(VideoSample, self).__init__()
        self.video = video
        self.video_url = video_url
        self.cover_url = cover_url
        self.created_time = created_time
        self.pl1_labels = pl1_labels
        self.pl2_labels = pl2_labels
        self.ml1_labels = ml1_labels
        self.ml2_labels = ml2_labels
        self.mvid = mvid

    def update(self, tag_map=None):
        if self.video is None and self.video_url is not None:
            self.video = os.path.split(self.video_url)[-1]

        if tag_map is not None:
            #  infer pl1 from pl2
            if self.pl1_labels is None and self.pl2_labels is not None:
                self.pl1_labels = map(lambda x: tag_map.pl2_to_pl1[x], self.pl2_labels)

            #  infer pl1 from ml1
            if self.pl1_labels is None and self.ml1_labels is not None:
                self.pl1_labels = map(lambda x: tag_map.ml1_to_pl1[x], self.ml1_labels)

            #  infer pl2 from ml2
            if self.pl2_labels is None and self.ml2_labels is not None:
                self.pl2_labels = map(lambda x: tag_map.ml2_to_pl2[x], self.ml2_labels)

            #  infer ml1 from pl1
            if self.ml1_labels is None and self.pl1_labels is not None:
                self.ml1_labels = map(lambda x: tag_map.pl1_to_ml1[x], self.pl1_labels)

            #  infer ml1 from pl2
            if self.ml1_labels is None and self.pl2_labels is not None:
                self.ml1_labels = map(lambda x: tag_map.ml2_to_ml1[tag_map.ml2_to_pl2[x]], self.pl2_labels)

            #  infer ml2 from pl2
            if self.ml2_labels is None and self.pl2_labels is not None:
                self.ml2_labels = map(lambda x: tag_map.pl2_to_ml2[x], self.pl2_labels)


class TagMap(object):
    def __init__(self):
        self.pl1_to_pl2 = {}
        self.pl2_to_pl1 = {}
        self.pl1_names = {}
        self.pl2_names = {}

        self.ml1_to_ml2 = {}
        self.ml2_to_ml1 = {}
        self.ml1_names = {}
        self.ml2_names = {}

        self.pl1_to_ml1 = {}
        self.pl2_to_ml2 = {}
        self.ml1_to_pl1 = {}
        self.ml2_to_pl2 = {}

        self.pl1_created_time = {}
        self.pl2_created_time = {}

    def load(self, filename):
        def _transform_key_to_int(_dict):
            return dict([(int(k), v) for k, v in _dict.iteritems()])

        with open(filename, 'r') as fd:
            s = json.loads(fd.read())
            self.pl1_names = _transform_key_to_int(s['pl1_names'])
            self.pl2_names = _transform_key_to_int(s['pl2_names'])
            self.pl1_to_pl2 = _transform_key_to_int(s['pl1_to_pl2'])
            self.pl2_to_pl1 = _transform_key_to_int(s['pl2_to_pl1'])
            self.pl1_to_ml1 = _transform_key_to_int(s['pl1_to_ml1'])
            self.pl2_to_ml2 = _transform_key_to_int(s['pl2_to_ml2'])
            self.pl1_created_time = _transform_key_to_int(s['pl1_created_time'])
            self.pl2_created_time = _transform_key_to_int(s['pl2_created_time'])
            self._infer_mtags()

    def save(self, filename):
        with open(filename, 'w') as fd:
            s = json.dumps({
                'pl1_names': self.pl1_names,
                'pl2_names': self.pl2_names,
                'pl1_to_pl2': self.pl1_to_pl2,
                'pl2_to_pl1': self.pl2_to_pl1,
                'pl1_to_ml1': self.pl1_to_ml1,
                'pl2_to_ml2': self.pl2_to_ml2,
                'pl1_created_time': self.pl1_created_time,
                'pl2_created_time': self.pl2_created_time,
            },
                indent=4,
                ensure_ascii=False,
                sort_keys=True)
            fd.write(s)

    def build_from_ptags(self, filename):
        for line in open(filename).readlines():
            t = line.strip().split()
            assert(len(t) == 3)
            l2, l1, name = int(t[0]), int(t[1]), t[2] # t[0] is level2, t[1] is level1

            if l1 == '0':
                # Level 1
                # l2 is the true label
                self.pl1_names[l2] = name
                self.pl1_to_pl2[l2] = set()
            else:
                # Level 2
                if l1 not in self.pl1_to_pl2:
                    self.pl1_to_pl2[l1] = set()
                self.pl1_to_pl2[l1].add(l2)

                if l2 not in self.pl2_to_pl1:
                    self.pl2_to_pl1[l2] = l1
                else:
                    assert(self.pl2_to_pl1[l2] == l1)

                if l2 not in self.pl2_names:
                    self.pl2_names[l2] = name
                else:
                    assert(self.pl2_names[l2] == name)

        l1_num = len(self.pl1_to_pl2.keys())
        l2_num = len(self.pl2_to_pl1.keys())
        self.pl1_to_ml1 = dict(zip(self.pl1_to_pl2.keys(), range(l1_num)))
        self.pl2_to_ml2 = dict(zip(self.pl2_to_pl1.keys(), range(l2_num)))
        self._infer_mtags()

    def update_ptags(self, pl2_to_pl1, pl1_names, pl2_names, pl1_created_time, pl2_created_time):
        self.pl1_created_time.update(pl1_created_time)
        self.pl2_created_time.update(pl2_created_time)

        for pl2, pl1 in pl2_to_pl1.iteritems():
            self.pl2_to_pl1[pl2] = pl1
            if pl1 not in self.pl1_to_pl2:
                self.pl1_to_pl2[pl1] = []
            self.pl1_to_pl2[pl1].append(pl2)

        for pl1, name in pl1_names.iteritems():
            self.pl1_names[pl1] = name

        for pl2, name in pl2_names.iteritems():
            self.pl2_names[pl2] = name

        self._infer_mtags()

    def update_ptom(self, pl1_to_ml1, pl2_to_ml2):
        self.pl1_to_ml1.update(pl1_to_ml1)
        self.pl2_to_ml2.update(pl2_to_ml2)
        self._infer_mtags()

    def _infer_mtags(self):
        for pl1 in self.pl1_to_pl2.keys():
            if pl1 not in self.pl1_to_ml1:
                self.pl1_to_ml1[pl1] = len(self.pl1_to_ml1)
            self.ml1_names[self.pl1_to_ml1[pl1]] = self.pl1_names[pl1]

        for pl2 in self.pl2_to_pl1.keys():
            if pl2 not in self.pl2_to_ml2:
                self.pl2_to_ml2[pl2] = len(self.pl2_to_ml2)
            self.ml2_names[self.pl2_to_ml2[pl2]] = self.pl2_names[pl2]

        for pl1, pl2s in self.pl1_to_pl2.iteritems():
            for pl2 in pl2s:
                ml1 = self.pl1_to_ml1[pl1]
                ml2 = self.pl2_to_ml2[pl2]
                if ml1 not in self.ml1_to_ml2:
                    self.ml1_to_ml2[ml1] = []
                self.ml1_to_ml2[ml1].append(ml2)

        for pl2, pl1 in self.pl2_to_pl1.iteritems():
            ml1 = self.pl1_to_ml1[pl1]
            ml2 = self.pl2_to_ml2[pl2]
            self.ml2_to_ml1[ml2] = ml1

        for pl1, ml1 in self.pl1_to_ml1.iteritems():
            self.ml1_to_pl1[ml1] = pl1

        for pl2, ml2 in self.pl2_to_ml2.iteritems():
            self.ml2_to_pl2[ml2] = pl2

    def l1_num_classes(self):
        return len(self.ml1_names)

    def l2_num_classes(self):
        return len(self.ml2_names)

def get_cmd_from_pid(pid):
    cmd_file = '/proc/%d/cmdline' % pid
    if not os.path.exists(cmd_file):
        raise ValueError('%s doesn\'t exists!' % cmd_file)
    return os.popen('cat %s | xargs -0 echo' % cmd_file).read().strip()

#  info is the displayed message by executing "nvidia-smi"
def parse_gpu_usage_str(info):
    #  TODO
    gpus = {}
    is_gpu_list = True
    gpu_id = None
    lines = info.split('\n')
    k = 7
    while k < len(lines):
        l = lines[k]
        t = l.split()

        if len(t) == 0:
            is_gpu_list = False

        if is_gpu_list:
            try:
                _id = int(t[1])
                gpu_id = _id
                t2 = lines[k+1].split()
                used_gm = int(t2[-7][:-3])
                gm = int(t2[-5][:-3])
                usage = int(t2[-3][:-1])
                gpus[gpu_id] = {
                    'used_memory': used_gm,
                    'memory': gm,
                    'usage': usage,
                    'process': [],
                }
            except:
                pass
        else:
            try:
                _id = int(t[1])
                pid = int(t[2])
                cmd = get_cmd_from_pid(pid)
                gm_used = int(t[5][:-3])
                gpus[_id]['process'].append((pid, cmd, gm_used))
            except:
                pass

        k += 1

    return gpus

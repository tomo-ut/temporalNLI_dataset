import xml.etree.ElementTree as ET
from collections import defaultdict
import pickle
import os


for filename in [
    "",
    "_30",
    "_50",
    "_verb_extend",
    "_verb_super_extend",
    "_verb_hyper_extend",
    "_casefreq1000_wordfreq100",
    "_casefreq3000_wordfreq100",
    "_casefreq1000_wordfreq200",
    "_casefreq1000_wordfreq300"
]:
    if os.path.exists('./external_data/kyoto-univ-web-cf-2.0/cf_dict' + filename + '.pickle'):
        continue
    tree = ET.parse('./external_data/kyoto-univ-web-cf-2.0/extract_cf' + filename + '.xml')
    cf_dict = defaultdict(dict)
    root = tree.getroot()
    # with open('./external_data/kyoto-univ-web-cf-2.0/extract_temp.xml', 'w') as out:
    for entry in root:
        if entry.attrib["predtype"] != "動":
            continue
        for caseframe in entry:
            key = []
            arg_dict = {}
            for argument in caseframe:
                comp_list = []
                for component in argument:
                    comp_list.append(component.text)
                if comp_list != []:
                    key.append(argument.attrib["case"])
                    arg_dict[argument.attrib["case"]] = comp_list
            key.sort()
            key = ",".join(key)
            cf_dict[key][caseframe.attrib["id"]] = arg_dict

    with open('./external_data/kyoto-univ-web-cf-2.0/cf_dict' + filename + '.pickle', 'wb') as f:
        pickle.dump(cf_dict, f)

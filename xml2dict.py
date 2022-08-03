import xml.etree.ElementTree as ET
from collections import defaultdict
import pickle


tree = ET.parse('./external_data/kyoto-univ-web-cf-2.0/extract_cf.xml')
cf_dict = defaultdict(dict)
root = tree.getroot()
# with open('./external_data/kyoto-univ-web-cf-2.0/extract_temp.xml', 'w') as out:
for entry in root:
    for caseframe in entry:
        key = []
        arg_dict = {}
        for argument in caseframe:
            key.append(argument.attrib["case"])
            comp_list = []
            for component in argument:
                comp_list.append(component.text)
            arg_dict[argument.attrib["case"]] = comp_list
        key.sort()
        key = ",".join(key)
        cf_dict[key][caseframe.attrib["id"]] = arg_dict

with open('./external_data/kyoto-univ-web-cf-2.0/cf_dict.pickle', 'wb') as f:
    pickle.dump(cf_dict, f)

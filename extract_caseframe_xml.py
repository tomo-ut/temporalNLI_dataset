import xml.etree.ElementTree as ET

tree = ET.parse('./external_data/kyoto-univ-web-cf-2.0/kyoto-univ-web-cf-2.0.xml')
# tree = ET.parse('./external_data/kyoto-univ-web-cf-2.0/temp.xml')
root = tree.getroot()
threshold = 10000
sample = 10
# with open('./external_data/kyoto-univ-web-cf-2.0/extract_temp.xml', 'w') as out:
remove_entry = []
for entry in root:
    valid_entry = False
    for caseframe in entry:
        if int(caseframe.attrib["frequency"]) >= threshold:
            valid_entry = True
    if not valid_entry:
        remove_entry.append(entry)
for entry in remove_entry:
    root.remove(entry)

for entry in root:
    remove_caseframe = []
    for caseframe in entry:
        if int(caseframe.attrib["frequency"]) < threshold:
            remove_caseframe.append(caseframe)
    for caseframe in remove_caseframe:
        entry.remove(caseframe)
    for caseframe in entry:
        for argment in caseframe:
            remove_component = []
            for i, component in enumerate(argment):
                if i >= sample:
                    remove_component.append(component)
            for component in remove_component:
                argment.remove(component)

tree.write('./external_data/kyoto-univ-web-cf-2.0/extract_cf.xml', encoding='utf-8')

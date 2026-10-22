"""Check the actual final media, test reports and source ZIP manifest."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile


def main():
    root=Path('output/submission')
    run=Path(Path('output/latest_run.txt').read_text().strip())
    audit=json.loads((run/'audit.json').read_text())
    assert audit['passed']
    junit=ET.parse('output/logs/unit_tests.xml').getroot()
    suites=list(junit.iter('testsuite'))
    count=sum(int(x.attrib.get('tests',0)) for x in suites)
    assert count>=19 and all(int(x.attrib.get('failures',0))==int(x.attrib.get('errors',0))==0 for x in suites)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
                      'format=duration:stream=codec_name,width,height','-of','json',str(root/'inspection_demo.mp4')]))
    duration=float(probe['format']['duration'])
    assert 60<=duration<=180
    assert probe['streams'][0]['codec_name']=='h264'
    assert (root/'report.pdf').read_bytes().startswith(b'%PDF')
    markdown=(root/'report.md').read_text(encoding='utf-8')
    diagram=Path('docs/system_architecture.mmd').read_text(encoding='utf-8').strip()
    assert '```mermaid\n'+diagram+'\n```' in markdown
    assert '概要と目的' in markdown and 'システム構成図' in markdown
    assert json.loads((root/'video_metadata.json').read_text())['presentation_language']=='ja'
    subprocess.run(['ffmpeg','-v','error','-i',str(root/'inspection_demo.mp4'),'-f','null','-'],check=True)
    with zipfile.ZipFile(root/'physicalai_source_and_evidence.zip') as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('MANIFEST.sha256.json'))
        assert all(hashlib.sha256(z.read(name)).hexdigest()==digest for name,digest in manifest.items())
        assert 'inspection/capture.py' in z.namelist()
    results={'status':'functional_demonstration_verified','audit':audit,'unit_tests_passed':count,
             'video_seconds':duration,'h264_decode':'passed','zip_manifest':'passed',
             'presentation_language':'ja',
             'known_limitations':['Raw classification accuracy is insufficient for practical inspection.',
                                  'Synthetic development inputs are not a held-out industrial test set.',
                                  'WSL host is Ubuntu 22.04; application containers are Ubuntu 24.04.',
                                  'No GitHub publication, YouTube upload, or OmniCampus submission performed.'],
             'artifacts':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in root.iterdir() if p.suffix in ('.mp4','.pdf','.zip','.md')}}
    (root/'completion.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()

"""Ephemeral OOXML fixtures exercise the application reader, not authored workbooks."""
import io
import zipfile
from xml.sax.saxutils import escape
from types import SimpleNamespace
import pytest
from streamlit.testing.v1 import AppTest
from services.candidate_input import read_uploads,prepare_input,run_prepared
from services.candidate_input_examples import mixed_test_tables,mixed_test_metadata,csv_test_files


def xlsx_fixture(formula=False):
    content=['<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
    book=[];rels=[];sheets={}
    for i,(name,frame) in enumerate(mixed_test_tables().items(),1):
        content.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        book.append(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>')
        rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>')
        rows=[]
        for r,values in enumerate([list(frame.columns),*frame.values.tolist()],1):
            cells=[]
            for c,value in enumerate(values,1):
                n=c;col=''
                while n: n,k=divmod(n-1,26);col=chr(65+k)+col
                coord=f'{col}{r}'
                if formula and i==2 and r==2 and c==7:
                    cells.append(f'<c r="{coord}"><f>1+1</f><v>2</v></c>')
                else: cells.append(f'<c r="{coord}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
            rows.append(f'<row r="{r}">{"".join(cells)}</row>')
        sheets[f'xl/worksheets/sheet{i}.xml']=f'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{"".join(rows)}</sheetData></worksheet>'
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w') as z:
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'+''.join(content)+'</Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml','<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'+''.join(book)+'</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(rels)+'</Relationships>')
        for path,xml in sheets.items():z.writestr(path,xml)
    return payload.getvalue()


def test_xlsx_matches_csv_result():
    p=prepare_input(read_uploads({'test.xlsx':xlsx_fixture()}),mixed_test_metadata())
    assert p.ready,p.issues
    assert run_prepared(p)['expected_net_value']==pytest.approx(74)


def test_cached_formula_rejected_not_silently_used():
    b=read_uploads({'test.xlsx':xlsx_fixture(formula=True)})
    assert any('수식 결과의 최신' in i['message_ko'] for i in b.issues)
    assert not prepare_input(b,mixed_test_metadata()).ready


@pytest.mark.parametrize('xlsx',[False,True])
def test_uploaded_page_run_search_bad_input_preserves_previous_history_download(monkeypatch,tmp_path,xlsx):
    import streamlit as st
    import pages.candidate_review as page
    monkeypatch.setenv('VARO_CANDIDATE_REVIEW','1')
    monkeypatch.setattr(page,'HISTORY_DIR',tmp_path)
    files={'test.xlsx':xlsx_fixture()} if xlsx else csv_test_files()
    objects=[SimpleNamespace(name=k,getvalue=lambda value=v:value) for k,v in files.items()]
    monkeypatch.setattr(st,'file_uploader',lambda *args,**kwargs:objects)
    at=AppTest.from_string('from pages.candidate_review import render_candidate_review_page\nrender_candidate_review_page()',default_timeout=30)
    at.session_state['candidate_data_mode']='실제 자료 검토'
    at.session_state['candidate_dataset']='EXPLICIT_TEST'
    at.session_state['candidate_network']='test-network'
    from datetime import date
    at.session_state['candidate_as_of']=date(2026,10,16)
    at.session_state['candidate_input_kind']='실제 자료 + 시험 가정'
    at.session_state['candidate_consent']=True
    at.run(); assert not at.exception
    at.button(key='candidate_validate_upload').click().run();assert not at.exception
    at.button(key='candidate_real_run').click().run();assert not at.exception
    before=at.session_state['_candidate_uploaded_result']
    assert before['expected_net_value']==pytest.approx(74)
    at.text_input(key='candidate_uploaded_search').set_value('P').run()
    assert len(at.get('download_button'))==1
    at.button(key='candidate_uploaded_save').click().run()
    assert len(list(tmp_path.glob('candidate_*.json')))==1
    objects[:]=[SimpleNamespace(name='bad.xlsx',getvalue=lambda:b'invalid')]
    at.button(key='candidate_validate_upload').click().run();assert not at.exception
    assert at.session_state['_candidate_uploaded_result']==before
    assert at.session_state['candidate_uploaded_search']=='P'
    assert any('이전 분석 결과는 그대로 유지' in w.value for w in at.warning)
    visible=' '.join(str(x.value) for kind in (at.markdown,at.warning,at.error,at.info) for x in kind)
    assert 'Traceback' not in visible and 'OPTIMAL' not in visible

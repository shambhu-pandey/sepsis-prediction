import sys
p='app.py'
try:
    s=open(p,'r',encoding='utf-8').read()
    compile(s,p,'exec')
    print('OK')
except SyntaxError as e:
    print('SYNTAX_ERROR', e.msg, 'line', e.lineno)
    # print context
    import traceback
    traceback.print_exc()
except Exception as e:
    print('ERROR', e)

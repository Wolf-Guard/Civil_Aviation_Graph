import os

#获取目录名
def text_process():
    paths = []
    count = 0
    filepath = 'Initial_text'
    name = os.listdir(filepath)
    for file in name:
        text_path = 'Initial_text/' + file
        paths.append(text_path)
    for path in paths:
        count +=1
        targetpath = 'processing_text/'+str(count)+'.txt'
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # print(len(lines))
            text = ''
            for line in lines:
                text += line.strip()
            text1 = text.split()
            text2 = ''.join(text1)
            # print(text2)
            text3 = text2.split('。')
            print(text3)

        with open(targetpath, 'w', encoding='utf-8') as fw:
            for line in text3:
                fw.write(line + "\n")
        fw.close()
        f.close()



def text_process2():
    paths = []
    count = 0
    filepath = 'processing_text'
    name = os.listdir(filepath)
    for file in name:
        text_path = 'processing_text/' + file
        paths.append(text_path)
    for path in paths:
        count +=1
        targetpath = 'processing2_text/'+str(count)+'.txt'
        with open(path, 'r', encoding='utf-8') as f:
            rubbish = []
            lines = f.readlines()
            print(lines)
            for num in range(len(lines)):
                a = ''
                b = ''
                c = ''
                d = ''
                e = ''
                f = ''
                g = ''
                h = ''
                I = ''
                j = ''
                k = ''
                if lines[num][40:50] == '..........':
                    a = num
                    # print(a)
                if a != '':
                    rubbish.append(lines[a])
                if lines[num][3:4] == '图':
                    b = num
                    # print(b)
                if b != '':
                    rubbish.append(lines[b])
                if lines[num][0:1] == '图':
                    c = num
                    # print(c)
                if c != '':
                    rubbish.append(lines[c])
                if lines[num][24:25] == '图':
                    d = num
                    # print(d)
                if d != '':
                    rubbish.append(lines[d])
                if lines[num][7:8] == '图':
                    e = num
                    # print(e)
                if e != '':
                    rubbish.append(lines[e])
                if lines[num][0:1] == '表':
                    f = num
                    # print(f)
                if f != '':
                    rubbish.append(lines[f])
                if lines[num][0:1] == '[':
                    g = num
                    # print(g)
                if g != '':
                    rubbish.append(lines[g])
                if lines[num][3:4] == '[':
                    h = num
                    # print(h)
                if h != '':
                    rubbish.append(lines[h])
                if lines[num][1:2] == '图':
                    I = num
                    # print(I)
                if I != '':
                    rubbish.append(lines[I])
                if lines[num][27:28] == '图':
                    j = num
                    # print(j)
                if j != '':
                    rubbish.append(lines[j])
                if lines[num][31:32] == '图':
                    k = num
                    # print(k)
                if k != '':
                    rubbish.append(lines[k])

            # print(rubbish)
            data = []
            for line in lines:
                if line not in rubbish:
                    data.append(line)
            result = ''.join(data)
            # print(result)
            with open(targetpath,'w',encoding='utf-8') as fw:
                fw.write(result)
            fw.close()
        f.close()

if __name__ == '__main__':
    text_process()
    text_process2()


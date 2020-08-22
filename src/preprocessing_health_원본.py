# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:56:52 2019

@author: Win10
"""
import re

def get_clean_comments(clean_comments, word2index_keys, FNAME="clean.xlsx", store=False, cnt=0):
    newmessage = []
    newmessage2 = []
    newmessage3 = []
    newUSER_ID = []
    newCNSL_SN = []
    n_comments = clean_comments.shape[0]
    cnt = 0
    for i in range(n_comments):
        tmp = clean_comments.message.values[i].split('\n')
        for message in tmp:
            if message == '' or message == ' ' or message == '  ' or message == " - " or message == ":":
                continue
            if len(message) <= 5:
                continue
            message = message.strip()
            message = re.sub("^- ", "", message)
            message = re.sub("^-", "", message)
            newsentence = re.sub("( ){2,}", " ", message)
            newmessage.append(newsentence)
            words, indices = list_encoding(newsentence, word2index_keys)
            newmessage2.append(words)
            newmessage3.append(indices)
            newUSER_ID.append(clean_comments.USER_ID[i])
            newCNSL_SN.append(clean_comments.CNSL_SN[i])
            cnt += 1
    
    clean_sentences = pd.DataFrame(list(zip(newUSER_ID, newCNSL_SN, newmessage, newmessage2, newmessage3)), \
                                   columns=['USER_ID', 'CNSL_SN', 'message', 'message2', 'message3'])
    if store:
        clean_sentences.to_excel(PATH + "./word2vec/" + FNAME, sheet_name = 'Sheet1', \
                             na_rep = 'NaN', float_format = "%.2f", 
                             header = True, #columns = ["group", "value_1", "value_2"], # if header is False
                             index = True, index_label = "num", startrow = 0, startcol = 0) 
    return clean_sentences, cnt

def get_tokens(comments, tokenizer):
    mess = list()
    mess_v = list()
    for s in comments:
        for token, tag in tokenizer.pos(s.replace(' ', '')):
            #if tag=='Noun' or tag == 'Verb' or tag == 'Adjective':
            if tag=='NNG' or tag == 'NNP':
                mess.append(token)
            elif tag == 'VV':
                mess_v.append(token)
                #elif tag == 'SN':
                #    mess.append(token)
    return mess, mess_v
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext
#cleanhtml(xx)
def preprocessing_health_j(xx):
    #&lt;/span&gt;&lt;span style=&quot;font-size: 10pt;&quot;&gt;
    #xx = re.sub("^ ^", "", xx)
    ss = xx
    idx1 = ss.find("img style=")
    idx2 = ss.find(".png")
    xx = ss[:idx1] + ss[(idx2+4):]

    #print(xx)    
    ss = xx
    xx = re.sub("<p style=", "", xx)

    xx = re.sub("img", "", xx)
    xx = re.sub("title=", "", xx)
    xx = re.sub("alt=", "", xx)
    xx = re.sub("src=", "", xx)
    xx = re.sub("khealth", "", xx)
    xx = re.sub("kr8089", "", xx)
    xx = re.sub("do", "", xx)
    xx = re.sub("attchFilesSn", "", xx)
    xx = re.sub("cmmn", "", xx)
    xx = re.sub("attchFileSn", "", xx)
    xx = re.sub("374541", "", xx)
    xx = re.sub("httpmhc", "", xx)
    xx = re.sub("attchFileDownload", "", xx)
    xx = re.sub("attchFileDtlsSn", "", xx)
    xx = re.sub("\d\.", "", xx)
    xx = re.sub("\.\)", ")", xx)
    xx = re.sub("\.", "\n", xx)
    xx = re.sub("니다!", "니다\n", xx)
    xx = re.sub("니다", "니다\n", xx)
    xx = re.sub("세요", "세요\n", xx)
    xx = re.sub("까요", "까요\n", xx)
    xx = re.sub("겠어요", "겠어요\n", xx)
    #
    xx = re.sub("0pt", " ", xx)
    xx = re.sub("0&", " & ", xx)
    xx = re.sub("&*br;*", " ", xx)
    xx = re.sub("&lt;", " ", xx)
    xx = re.sub("&hearts", " ", xx)
    xx = re.sub("&amp", "", xx)
    xx = re.sub("&darr", "", xx)
    xx = re.sub("span", " ", xx)
    xx = re.sub("&gt;", " ", xx)
    xx = re.sub("span", " ", xx)
    xx = re.sub("stype", " ", xx)
    xx = re.sub("false,", " ", xx)
    xx = re.sub("&quot;", " ", xx)
    xx = re.sub("&ldquo", " ", xx)
    xx = re.sub("&lsquo", " ", xx)
    xx = re.sub("&rdquo", " ", xx)
    xx = re.sub("&rarr", " ", xx)
    xx = re.sub("&apos",  " ", xx)
    xx = re.sub("&sim", " ", xx)
    xx = re.sub("& lt", " ", xx)
    xx = re.sub("& quot", " ", xx)
    xx = re.sub(" -line ", "", xx)
    xx = re.sub("^&[a-zA-Z]",  "", xx)
    xx = re.sub("////", "", xx)
    xx = re.sub("class=0", "", xx)
    xx = re.sub("/p|/div", "", xx)
    xx = re.sub("\<b\>", "", xx)
    xx = re.sub("\<\/b\>", "", xx)
    xx = re.sub("\np|\ndiv", "", xx)
    xx = re.sub("\<u\>", "", xx)
    xx = re.sub("\</u\>", "", xx)
    xx = re.sub("SE3-TEXT", "", xx)
    xx = re.sub("d●▼●b", "", xx)
    xx = re.sub("\d{3,3}[\)|-]\d{3,}-\d{4,}", "", xx)
    
    #xx = re.sub("//span", "", xx)
    #xx = re.sub("&lt;p&gt;|&lt;|p&gt;|\n&lt;|br|&gt;", "", xx)
    xx = re.sub("style=|&nbsp;", "", xx)
    xx = re.sub("font-size: 10pt;", "", xx)
    #xx = re.sub("/spanspan ", "", xx)
    xx = re.sub("font-family: 함초롬바탕;", "", xx)
    xx = re.sub("letter-spacing:", "", xx)
    xx = re.sub("//span", "", xx)
    xx = re.sub("/span", "", xx)
    xx = re.sub("span", "", xx)
    xx = re.sub("/p class=0", "", xx)
    xx = re.sub("class=", "", xx)
    xx = re.sub("style=\"color", "", xx)
    xx = re.sub("se_fs_T\d", "", xx)
    xx = re.sub("//", "", xx)
    xx = re.sub("baseline;", "", xx)
    xx = re.sub("border: 0px solid rgb(0, 0, 0);", "", xx) 
    xx = re.sub("width: 18px;", "", xx)
    xx = re.sub("height: 160%", "", xx)
    xx = re.sub("height: 18px;", "", xx)
    xx = re.sub("vertical-align:", "", xx)
    xx = re.sub("border: 0px", "", xx)
    xx = re.sub("solid", "", xx)
    xx = re.sub("rgb(0, 0, 0);", "", xx)
    xx = re.sub("rgb\(244, 204, 204\)", "", xx)
    xx = re.sub("rgb\(\d{1,}\, \d{1,}, \d{1,}\)", "", xx)
    xx = re.sub("background-color: rgb(0, 255, 0);", "", xx)
    xx = re.sub("background-color: rgb(0, 255, 255);", "", xx)
    xx = re.sub("color: rgb(255, 0, 0);", "", xx)
    xx = re.sub("color: rgb(255, 0, 0);", "", xx)
    xx = re.sub("color: rgb(0, 255, 255);", "", xx)
    xx = re.sub("color: rgb(0, 255, 0);", "", xx)
    xx = re.sub("color:", "", xx)
    xx = re.sub("David;", "", xx)
    xx = re.sub("^(\.img\S)(.+)(;$)", "", xx)
    xx = re.sub("font-family: 한컴바탕;", " ", xx)
    xx = re.sub("font-family: 굴림;", " ", xx)
    xx = re.sub("font-family: 돋움;", " ", xx)
    xx = re.sub("font-family: 바탕", " ", xx)
    xx = re.sub("font-family: simsun", " ", xx)
    
    xx = re.sub("margin-right", "", xx)
    xx = re.sub("margin-left", "", xx)
    xx = re.sub("margin", "", xx)
    xx = re.sub(" word-eak keep-all", "", xx)
    xx = re.sub("line-height: \d{2,}\%", "", xx)
    xx = re.sub("line-height:", " ", xx)
    xx = re.sub("바탕글", "", xx)
    xx = re.sub("font-family:", " ", xx)
    xx = re.sub("table border-collapse:", " ", xx)
    xx = re.sub("font-weight: bold", " ", xx)
    xx = re.sub("font-size", " ", xx)
    xx = re.sub("collapse", " ", xx)
    xx = re.sub("border-width: initial", " ", xx)
    xx = re.sub("border-style:", " ", xx)
    xx = re.sub("border-color:", "", xx)
    xx = re.sub("border-image", "", xx)
    xx = re.sub("border-width:", " ", xx)
    xx = re.sub("td width:", " ", xx)
    xx = re.sub("width", " ", xx)
    xx = re.sub("border", " ", xx)
    xx = re.sub("td", " ", xx)
    
    xx = re.sub("height:", " ", xx)
    xx = re.sub("tbody", " ", xx)
    xx = re.sub("tr td", " ", xx)
    xx = re.sub("\d{1,}px", "", xx)
    xx = re.sub("\d{1,}pt", "", xx)
    xx = re.sub("padding:", "", xx)
    xx = re.sub("(rgb)", "", xx)
    xx = re.sub("(\(000\))", "", xx)
    xx = re.sub("00255", "", xx)
    xx = re.sub("바탕글(\s){1,}", " ", xx)
    xx = re.sub("text-align: center", " ", xx)
    xx = re.sub(" 150\%", " ", xx)
    xx = re.sub(" 180\%", " ", xx)
    xx = re.sub("//td", " ", xx)
    xx = re.sub("//table", "", xx)
    xx = re.sub("tr", "", xx)
    xx = re.sub("table", "", xx)
    xx = re.sub("collapse", " ", xx)
    xx = re.sub("\d{1,}\d{1,}pt", "", xx)
    xx = re.sub("\d{1,}\.\d{1,}pt", "", xx)
    xx = re.sub("margin-bottom:", "", xx)
    xx = re.sub("margin-top:", "", xx)
    xx = re.sub("StartFragment", "", xx)
    xx = re.sub("경과입니다", " 경과입니다 ", xx)
    xx = re.sub("id=.*", "---", xx)
    xx = re.sub("text-decoration", "", xx)
    #xx = re.sub("\d{1,}", "99", xx)
    #^.*찾는 문자열.*$rn
    # 그림 제거
    #xx = re.sub("(img src)(.+?)(\.png)", "", xx)
    #print(xx)
    xx = re.sub("함초롬바탕", "", xx)
    xx = re.sub("굴림", "", xx)
    #xx = re.sub()
    xx = re.sub("(img src)(.+?)(0\))", " ", xx)
    xx = re.sub("(img title)(.+?)(this\);)", " ", xx)
    xx = re.sub("font-size: 13.3333px;", " ", xx)
    xx = re.sub("\.{2,}", " ", xx)
    
    # 쉼표없애기
    xx = re.sub(", |,", "", xx)
    xx = re.sub("\!|\^|\?|\＾|\;|\★|\♪", " ", xx)
    #xx = re.sub("\\n", " ", xx)
    xx = re.sub("\ㅎ", " ", xx)
    xx = re.sub("\*|\ㅜ|\※|\♡|\♬|\♥", " ", xx)
    xx = re.sub("\$|\\t", " ", xx)
    #xx = re.sub("\~|\-")  <= 3 ~ 5회
    
    
    xx = re.sub("\ㄱ|\ㄴ|\ㄷ|\ㄹ|\ㅁ", " ", xx)
    #xx = re.sub("[0-9]{1,}", " 99 ", xx)
    #xx = re.sub("[0-9] [0-9] [0-9] [0-9]", [0-9][0-9][0-9][0-9], xx)
    
    #8,000: 숫자 네자리 이상 
    #g = re.findall("[0-9]+\,[0-9]{3}", " ", xx)
    #g.replace(',', '')
    
    xx = re.sub("background-image:", "", xx)
    xx = re.sub("background-position:", "", xx)
    xx = re.sub("background-size:", "", xx)
    xx = re.sub("background-repeat:", "", xx)
    xx = re.sub("background-origin:", "", xx)
    xx = re.sub("background-clip:", "", xx)
    xx = re.sub("background-attachment:", "", xx)
    xx = re.sub(" initial ", "", xx)
    xx = re.sub("lack", "", xx)
    xx = re.sub("background-", "", xx)
    xx = re.sub("background:", "", xx)
    xx = re.sub("strong", "", xx)
    xx = re.sub("255255255", "", xx)
    xx = re.sub("background-c", "", xx)
    xx = re.sub("text-decoration-line", "", xx)
    xx = re.sub("underline", "", xx)
    xx = re.sub(" (02550) ", "", xx)
    #print(xx)
    xx = re.sub("(background).*$:", "", xx)
    xx = re.sub("맑은 고딕", "", xx)
    xx = re.sub(":", "", xx)
    #xx = re.sub("\n", " ", xx)
    xx = re.sub("\t", " ", xx)
    xx = re.sub("~", " ", xx)
    xx = re.sub("'515151'", " ", xx)
    xx = re.sub("  1  ", " ", xx)
    
    xx = re.sub("p안녕", "안녕", xx)
    xx = re.sub("text-align justify ", "", xx)
    xx = re.sub("align= left          140%", "", xx)
    xx = re.sub("text-align justify ", "", xx)
    xx = re.sub("140%  word-eak keep-all ", "", xx)
    xx = re.sub("align= left", "", xx)
    xx = re.sub("serif", "", xx)
    xx = re.sub("b       blue    /bb       blue", "", xx)
    xx = re.sub("p  ", "", xx)
    xx = re.sub("  120%  ", "", xx)
    xx = re.sub("normal  word-eak keep-all", "", xx)
    xx = re.sub(" text-align left", "", xx)
    xx = re.sub("----", "", xx)
    xx = re.sub("-ms-layout-grid-mode both", "", xx)
    xx = re.sub("2552550", "", xx)
    xx = re.sub("lang= EN-US ", "", xx)
    xx = re.sub("0255255", "", xx)
    xx = re.sub("25500", "", xx)
    xx = re.sub("02550", "", xx)
    xx = re.sub("00300", "", xx)
    xx = re.sub("\s\(00255\)\s", "", xx)
    xx = re.sub("\s\(25500\)\s", "", xx)
    xx = re.sub("\s\(10616879\)\s", "", xx)
    xx = re.sub("\s\(000\)\s", "", xx)
    xx = re.sub("alt= 방긋", "", xx)
    xx = re.sub(" none ", "", xx)
    xx = re.sub(" img  ", "", xx)
    xx = re.sub("\(\)", "", xx)
    xx = re.sub("13\.(\s){1,}13\.", "", xx)
    xx = re.sub("9\.(\s){1,}9\.", "", xx)
    xx = re.sub("9\.(\s){1,}\.", "", xx)
    
    ##
    xx = re.sub("httpmhc", "", xx)  
    xx = re.sub("httpsmhc", "", xx)
    xx = re.sub("kr8089", "", xx)
    xx = re.sub("& lt", "", xx)
    xx = re.sub("/", "", xx)
    xx = re.sub(" p ", "", xx)
    xx = re.sub("=445059", "", xx)
    xx = re.sub("=1 ", "", xx)
    xx = re.sub("or", "", xx)
    # 특수문자 제거 
    xx = re.sub("p5차 ", "5차 ", xx)
    xx = re.sub("p대상자", "대상자", xx)
    xx = re.sub("rarr", "", xx)
    #xx = re.sub("\W+", " ", xx)
    xx = re.sub(" {2,}", " ", xx)
    
    ##
    xx = re.sub("(^jav)[a-zA-Z\(\)]{1,}", " ", xx)
    #xx = re.sub("(^krc)[a-zA-Z\d]{1,}", " ", xx)
    xx = re.sub("krcrosseditimagesemoticon3png", "", xx)
    xx = re.sub("3png", "", xx)
    xx = re.sub("krcrosseditimagesemoticon2png", "", xx)
    xx = re.sub("krcrosseditimagesemoticon0png", "", xx)
    xx = re.sub("b ", "", xx)
    xx = re.sub("u ", "", xx)
    xx = re.sub("red ", "", xx)
    xx = re.sub("blue ", "", xx)
    xx = re.sub("song ", "", xx)
    xx = re.sub("div ", "", xx)
    xx = re.sub("(\d){4,6}", "", xx)
    xx = re.sub("onclick", "", xx)
    xx = re.sub("amp", "", xx)
    xx = re.sub("&", "", xx)
    xx = re.sub("=", "", xx)
    xx = re.sub("left to ", "", xx)
    xx = re.sub("javascriptinAppBrowserImageView", "", xx)
    xx = re.sub("\(this\)", "", xx)
    xx = re.sub("nmal wd- eak keep-all black", "", xx)
    xx = re.sub("song font-weight nmal", "", xx)
    xx = re.sub("wd- eak keep-all", "", xx)
    xx = re.sub("\(0\)", "", xx)
    xx = re.sub(" a href", "", xx)
    xx = re.sub("httpu-health", "", xx)
    xx = re.sub(" bong ", "", xx)
    xx = re.sub(" go ", "", xx)
    xx = re.sub(" krhcalmetabolism", "", xx)
    xx = re.sub(" ashttpu-health", "", xx)
    xx = re.sub(" krhcalmetabolism", "", xx)
    xx = re.sub("asp a", "", xx)
    xx = re.sub(" as", "", xx)
    xx = re.sub(" httpwww", "", xx)
    xx = re.sub("buk", "", xx)
    xx = re.sub("daegu", "", xx)
    xx = re.sub("www", "", xx)
    xx = re.sub("^\s", "", xx)
    xx = re.sub("-line", "", xx)
    #xx = re.sub("？", "", xx)
    xx = re.sub(" b", "", xx)
    xx = re.sub("\(5\)", "", xx)
    xx = re.sub("ong", "", xx)
    xx = re.sub("go ", "", xx)
    xx = re.sub("[\(|\)]", "", xx)
    xx = re.sub("[\[|\]]", " ", xx)
    xx = re.sub(" # ", "", xx)
    xx = re.sub("font-variant-numeric", "", xx)
    xx = re.sub("nmal ", "", xx)
    xx = re.sub("font-variant-east-asian", "", xx)
    xx = re.sub(" -to", "", xx)
    xx = re.sub("-bottom", "", xx)
    xx = re.sub(" font-weight 400", "", xx)
    xx = re.sub("overflow-wraeak-wd font-variant-ligatures font-variant-caps phans 2 text-align start wiws 2 -webkit-text-soke- -style - wd-spacing", "", xx)
    xx = re.sub("spacing", "", xx)
    xx = re.sub("ㅠ", "", xx)
    xx = re.sub("_ _", "", xx)
    xx = re.sub(" ---", "", xx)
    xx = re.sub("li", "", xx)
    xx = re.sub("ui", "", xx)
    xx = re.sub("[@|:|○]", "", xx)
    xx = re.sub("앞 으 로 도 꾸 준 히 주 5 일 이 상 8 보 이 상 걸 어 주 세 요", "앞으로도 꾸준히 주 5일 이상 8보 이상 걸어주세요", xx)

    ### 건강추가
    xx = re.sub("\"", "", xx)
    xx = re.sub("HStyle0", "", xx)
    xx = re.sub("\r", "", xx)
    xx = re.sub("\"\d\"", "", xx)
    xx = re.sub("(^---\nhwpta).{1,}", "", xx)
    xx = re.sub("\np", "", xx)
    xx = re.sub("langEN-US", "", xx)
    xx = re.sub("-left-", "", xx)
    xx = re.sub("-right-", "", xx)
    xx = re.sub("-webkit-print-col-adjust", "", xx)
    xx = re.sub("NamoSEshow", "", xx)
    xx = re.sub("p-", "", xx)
    xx = re.sub("exact", "", xx)
    xx = re.sub("Malgun Gothictum", "", xx)
    xx = re.sub("돋움", "", xx)
    xx = re.sub("NamoSEshow", "", xx)
    
    ### 영양추가
    xx = re.sub("<p>", "", xx)
    xx = re.sub("<p >", "", xx)
    xx = re.sub("[<|>]", "", xx)
    xx = re.sub("\r", "", xx)
    xx = re.sub("\"", "", xx)
    xx = re.sub("122 Malgun Gothictum돋움", "", xx)
    xx = re.sub("GumarialTahomaVerdanasans", "", xx)
    xx = re.sub("[◆|▶|☆|●]", "", xx)
    xx = re.sub("＃", "", xx)
    xx = re.sub("ne-", "", xx)
    xx = re.sub("rsquo", "", xx)
    xx = re.sub("div", "", xx)
    xx = re.sub("SanBsB-Identity-H", "", xx)
    ss = xx
    idx1 = ss.find("hwpta")
    idx2 = ss.find("}--")
    ss = ss[:idx1] + ss[idx2:]
    
    idx1 = ss.find("hwpta")  
    idx2 = ss.find("id 1ah 0av 0ht 0hi ")
    ss = ss[:idx1] + ss[idx2:]

    idx1 = ss.find("hwpta")  
    idx2 = ss.find("-4lt 1lw 1lc 0sa 850sb 567st")
    ss = ss[:idx1] + ss[idx2:]

    idx1 = ss.find("hwpta")  
    idx2 = ss.find("01D550E93AF2CBbf 0ru { cp 0")
    ss = ss[:idx1] + ss[idx2:]

    xx = ss

    xx = re.sub("hwp_edit_board_content", "", xx)
    xx = re.sub("160% \np", "", xx)
    xx = re.sub("text-agn left", "", xx)
    xx = re.sub("gtng", "", xx)
    xx = re.sub("____", "", xx)
    xx = re.sub("text-indent", "", xx)
    xx = re.sub("\np 0", "", xx)
    xx = re.sub("\np", "", xx)
    xx = re.sub(" 궁서 ", "", xx)
    xx = re.sub("style", "", xx)
    xx = re.sub("s 204", "", xx)
    
    xx = re.sub("\n( ){1,}\n", "\n", xx)
    xx = re.sub("\n( ){1,}\n", "\n", xx)
    xx = re.sub("\n( ){1,}\n", "\n", xx)
    xx = re.sub("\n\n", "\n", xx)
    xx = re.sub("\n\n", "\n", xx)
    xx = re.sub("\n\n", "\n", xx)
    xx = re.sub("[ ]{1,}", " ", xx)
    #xx = re.sub("\W", "", xx)
    xx = re.sub("p 0 # ", "", xx)
    xx = re.sub("0 # ", "", xx)
    xx = re.sub("p 0 ", "", xx)                
    xx = re.sub(" 0 ", "", xx)
    xx = re.sub(" s", "", xx)
    ss = xx
    idx1 = ss.find("--")  
    idx2 = ss.find("\} \}--")
    xx = ss[:idx1] + ss[idx2:]
    xx = re.sub("\}a", "", xx)
    
    
    xx = xx.strip()
    #xx = re.sub("[☆|ㅠ|#|▶|①|②|③|④|⑥|⑥", "", xx)
    return xx
#



def isNaN(num):
    return num != num

def preprocessing_health(x):
    #xx = [a.lower() for a in x]
    #xx = [re.sub("[가-힝]+[고객]*(님{1})", " CUSTOMER ", a) for a in xx]
    x_clean = []
    print('Lenght of input strings={}'.format(len(x)))
    for i in range(len(x)):
        if i % 5000 == 0:
            print('{} th string'.format(i))
        if isNaN(x[i]) != True:
            x_clean.append(preprocessing_health_j(x[i]))
        else:
            x_clean.append('nan')
    #xx
    #yy = re.sub("(span0|span\\;|&lt|&quot|&gt|&ldquo|&rdquo|letter-spacing)", " ", xx)
    #yy = re.sub("StartFragment|style=|font-family:|layout-grid-mode:|both;", " ", yy)
    #yy = re.sub("\W", " ", yy)
    #yy = re.sub("(span|font size  10pt)", "", yy)
    #yy = re.sub("stylequotfontfamily바탕quotgt", " ", yy)
    return x_clean

def preprocessing_health2(x):
    xx = [a.lower() for a in x]
    xx = [re.sub("[가-힝]+[고객]*(님{1})", " CUSTOMER ", a) for a in xx]
    xx = [re.sub("[가-힝]+\\s{1,}고객", " CUSTOMER ", a) for a in xx]

    #[re.sub("(,[ ]*!.*)$", "", a) for a in xx]
    ####
    xx = [re.sub("\\[cj[ ]*헬로(비전|비젼)*\\]", "", a) for a in xx]
    
    # 은행
    #xx <- "농협-999-999999-99-999"
    # 농협999999 -99-999999
    #xx = [re.sub("농협[\\s-]*9{3}-*9{2,}-*9{2,}-*9{2,}", " ACCOUNT ", a) for a in xx]
    xx = [re.sub("농협(.|\\s)*9{3}-*9{2,}-*9{2,}-*9{2,}", " ACCOUNT ", a) for a in xx]
    xx = [re.sub("농협.9{3,}(-|\\s)*9{2}-*9{3,}", " ACCOUNT ", a) for a in xx]
    #농협 999999-99-999999
    xx = [re.sub("농협(\\s)*9{3,}-*9{2,}-*9{3,}", " ACCOUNT ", a) for a in xx]
    xx = [re.sub("신한[\\s-]*9{3}-*9{2,}-*9{2,}-*9{3,}", " ACCOUNT ", a) for a in xx]
    xx = [re.sub("신한\\s*9{3,}-*9{2,}-*9{3,}", " ACCOUNT ", a) for a in xx]
    #국민-999-999999-99-999
    xx = [re.sub("국민(\\s|-)*9{3,}-*9{2,}-*9{2,}-*9{2,}", " ACCOUNT ", a) for a in xx]

    return xx
"""
def remove_letter():  # 문자열에서 선택된 특정 문자를 없애버리기
    string_length = len(xx)
    location = 0

    while (location < string_length) :
        if base_string[location] == letter_remove:
            base_string = base_string[:location] + base_string[location+1::]  # [:a] -> 처음부터 a위치까지, [a::]a위치부터 끝
            string_length = len(base_string)
        location+= 1
    print "Result: %s" % base_string
    return
"""
#
# 2019/10/20
#    
def preprocessing_health3(x):
    xx = [a.lower() for a in x]
    xx = [re.sub("[가-힝]+[고객]*(님{1})", " CUSTOMER ", a) for a in xx]
    xx = [re.sub("[가-힝]+\\s{1,}고객", " CUSTOMER ", a) for a in xx]
    #xx = [re.sub("[가-힝]+[고객]*(님{1})", " ", a) for a in xx]
    #xx = [re.sub("[가-힝]+\\s{1,}고객", " ", a) for a in xx]
    
    xx = [re.sub("(광고)", "", a) for a in xx]
    xx = [re.sub("cj", "", a) for a in xx]
    xx = [re.sub("\[", "", a) for a in xx]
    xx = [re.sub("\]", "", a) for a in xx]
    xx = [re.sub("무료수신거부", "", a) for a in xx]
    
    ## 주소
    ##로 99
    xx = [re.sub("(로|길|골|동)\\s[0-9]{1,}", " ADDRESS ", a) for a in xx]
    ##9층
    xx = [re.sub("[0-9]{1,}\\s*층", " FLOOR ", a) for a in xx]
    xx = [re.sub("(길|골|동)(.|[0-9]{1,})(길|동|.)*[0-9]{1,}", " ADDRESS ", a) for a in xx]
    xx = [re.sub("로[0-9]{1,}(번|길)*(번길)*[0-9]-*", " ADDRESS ", a) for a in xx]
    xx = [re.sub("([0-9]{1,}-[0-9]{1,}번지)(.)", " ADDRESS ", a) for a in xx]
    xx = [re.sub("리(\\s)*([0-9]{1,}-[0-9]{1,})(\\s)*", " ADDRESS ", a) for a in xx]
    xx = [re.sub("[0-9]{1,}(로|번|길)(\\s)*([0-9]{1,}-[0-9]{1,})*", " ADDRESS ", a) for a in xx]
    #9로 9-99
    
    ####
    #금액
    xx = [re.sub("9{1,}(\\,|\\.)*9{3}([ ])*원", " CHARGE ", a) for a in xx]
    xx = [re.sub("9{2,}원", " CHARGE", a) for a in xx]
    xx = [re.sub("9{2,}\\,9{3}코인", " CHARGE", a) for a in xx]
 
    # 팩스
    xx = [re.sub("팩스\\s9{3,}\\s9{2,}\\s9{3,}", " FAX ", a) for a in xx]
    ## 9숫자 11개
    xx = [re.sub("9{8,}", " PHONE", a) for a in xx]
    xx = [re.sub("9{3,}[\\s-]9{3,}[\\s-]9{3,}", " PHONE ", a) for a in xx]
    xx = [re.sub("9{3,}\\s{1,}9{3,}", " PHONE ", a) for a in xx]
    xx = [re.sub("9{3,}-9{3,}", " PHONE ", a) for a in xx]
    xx = [re.sub("9{2,} 9{3,} 9{3,}", " PHONE ", a) for a in xx]
    xx = [re.sub("9{2,}[\\.-]9{3,}[\\.-]9{3,}", " PHONE ", a) for a in xx]
  
    xx = [re.sub("9{1,}일이내", " LIMITDAY ", a) for a in xx]
    xx = [re.sub("9{1,}[개]*[월일]", " MONTH ", a) for a in xx]

    ## 시간
    xx = [re.sub("[0-9]{1,}시", " HOUR ", a) for a in xx]
    xx = [re.sub("[0-9]{1,}분", " MINITUE ", a) for a in xx]
    ## 년도
    xx = [re.sub("\\s*9{4}년[\\s]*", " YEAR ", a) for a in xx]
    xx = [re.sub("m.mycatchon.com", " HOMEPAGE ", a) for a in xx]
  
  
    ####
    xx = [re.sub("납부안 내", " 납부 안내 ", a) for a in xx]
    xx = [re.sub("해지", " 해지 ", a) for a in xx]
    xx = [re.sub("cj헬로tv", " cj헬로tv ", a) for a in xx]
    xx = [re.sub("부천방송", " 부천방송 ", a) for a in xx]
    xx = [re.sub("나라방송", " 나라방송 ", a) for a in xx]
    xx = [re.sub("입니다", " ", a) for a in xx]
    xx = [re.sub("합니다", " ", a) for a in xx]
    xx = [re.sub("되었습니다", " ", a) for a in xx]
    xx = [re.sub("헬로", " ", a) for a in xx]
    xx = [re.sub("드립니다", " ", a) for a in xx]
    xx = [re.sub("드립겠습니", " 드립겠습니다 ", a) for a in xx]
    xx = [re.sub("부탁", " 부탁 ", a) for a in xx]
    xx = [re.sub("CJ", " CJ " , a) for a in xx]
    xx = [re.sub("AS", " AS " , a) for a in xx]
    xx = [re.sub("bank", " bank " , a) for a in xx]
    xx = [re.sub("CJONE", " CJONE " , a) for a in xx]
    xx = [re.sub("count", " count " , a) for a in xx]
    xx = [re.sub("CRM", " CRM " , a) for a in xx]
    xx = [re.sub("CS", " CS " , a) for a in xx]
    xx = [re.sub("day", " day " , a) for a in xx]
    xx = [re.sub("EMAIL", " EMAIL " , a) for a in xx]
    xx = [re.sub("fax", " fax " , a) for a in xx]
    xx = [re.sub("HELLO", " HELLO " , a) for a in xx]
    xx = [re.sub("Index", " Index " , a) for a in xx]
    xx = [re.sub("JTBC", " JTBC " , a) for a in xx]
    xx = [re.sub("kg", " kg " , a) for a in xx]
    xx = [re.sub("LED", " LED " , a) for a in xx]
    xx = [re.sub("LG", " LG " , a) for a in xx]
    xx = [re.sub("money", " money " , a) for a in xx]
    xx = [re.sub("month", " month " , a) for a in xx]
    xx = [re.sub("name", " name " , a) for a in xx]
    xx = [re.sub("PC", " PC " , a) for a in xx]
    xx = [re.sub("pw", " pw " , a) for a in xx]
    xx = [re.sub("Q LED", " QLED " , a) for a in xx]
    xx = [re.sub("Season", " Season " , a) for a in xx]
    xx = [re.sub("SPen", " SPen " , a) for a in xx]
    xx = [re.sub("stb", " stb " , a) for a in xx]
    xx = [re.sub("system", " system " , a) for a in xx]
    xx = [re.sub("TV", " TV " , a) for a in xx]
    xx = [re.sub("UHD", " UHD " , a) for a in xx]
    xx = [re.sub("VOD", " VOD " , a) for a in xx]
    xx = [re.sub("vod", " vod " , a) for a in xx]
    xx = [re.sub("www", " www " , a) for a in xx]
    xx = [re.sub("year", " year " , a) for a in xx]
    xx = [re.sub("가능", " 가능 " , a) for a in xx]
    xx = [re.sub("가야", " 가야 " , a) for a in xx]
    xx = [re.sub("가입", " 가입 " , a) for a in xx]
    xx = [re.sub("간편", " 간편 " , a) for a in xx]
    xx = [re.sub("감사", " 감사 " , a) for a in xx]
    xx = [re.sub("감사", " 감사 " , a) for a in xx]
    xx = [re.sub("강력", " 강력 " , a) for a in xx]
    xx = [re.sub("강원", " 강원 " , a) for a in xx]
    xx = [re.sub("개월", " 개월 " , a) for a in xx]
    xx = [re.sub("개인", " 개인 " , a) for a in xx]
    xx = [re.sub("개통", " 개통 " , a) for a in xx]
    xx = [re.sub("거부", " 거부 " , a) for a in xx]
    xx = [re.sub("건조", " 건조 " , a) for a in xx]
    xx = [re.sub("겨울", " 겨울 " , a) for a in xx]
    xx = [re.sub("결과", " 결과 " , a) for a in xx]
    xx = [re.sub("결제", " 결제 " , a) for a in xx]
    xx = [re.sub("결합", " 결합 " , a) for a in xx]
    xx = [re.sub("경남", " 경남 " , a) for a in xx]
    xx = [re.sub("경남", " 경남 " , a) for a in xx]
    xx = [re.sub("경로", " 경로 " , a) for a in xx]
    xx = [re.sub("경품", " 경품 " , a) for a in xx]
    xx = [re.sub("경험", " 경험 " , a) for a in xx]
    xx = [re.sub("계좌", " 계좌 " , a) for a in xx]
    xx = [re.sub("고객", " 고객 " , a) for a in xx]
    xx = [re.sub("고급", " 고급 " , a) for a in xx]
    xx = [re.sub("고속", " 고속 " , a) for a in xx]
    xx = [re.sub("공기", " 공기 " , a) for a in xx]
    xx = [re.sub("교체", " 교체 " , a) for a in xx]
    xx = [re.sub("교환", " 교환 " , a) for a in xx]
    xx = [re.sub("구매", " 구매 " , a) for a in xx]
    xx = [re.sub("구비", " 구비 " , a) for a in xx]
    xx = [re.sub("국내", " 국내 " , a) for a in xx]
    xx = [re.sub("궁금", " 궁금 " , a) for a in xx]
    xx = [re.sub("규모", " 규모 " , a) for a in xx]
    xx = [re.sub("금액", " 금액 " , a) for a in xx]
    xx = [re.sub("기간", " 기간 " , a) for a in xx]
    xx = [re.sub("기능", " 기능 " , a) for a in xx]
    xx = [re.sub("기대", " 기대 " , a) for a in xx]
    xx = [re.sub("기술", " 기술 " , a) for a in xx]
    xx = [re.sub("기준", " 기준 " , a) for a in xx]
    xx = [re.sub("기타", " 기타 " , a) for a in xx]
    xx = [re.sub("납부", " 납부 " , a) for a in xx]
    xx = [re.sub("내역", " 내역 " , a) for a in xx]
    xx = [re.sub("냉장고", " 냉장고 " , a) for a in xx]
    xx = [re.sub("노력", " 노력 " , a) for a in xx]
    xx = [re.sub("노트북", " 노트북 " , a) for a in xx]
    xx = [re.sub("녹화", " 녹화 " , a) for a in xx]
    xx = [re.sub("다시", " 다시 " , a) for a in xx]
    xx = [re.sub("다이렉트", " 다이렉트 " , a) for a in xx]
    xx = [re.sub("다이슨", " 다이슨 " , a) for a in xx]
    xx = [re.sub("단위", " 단위 " , a) for a in xx]
    xx = [re.sub("단체", " 단체 " , a) for a in xx]
    xx = [re.sub("담당", " 담당 " , a) for a in xx]
    xx = [re.sub("당월사용금", " 당월사용금 " , a) for a in xx]
    xx = [re.sub("당사", " 당사 " , a) for a in xx]
    xx = [re.sub("대구", " 대구 " , a) for a in xx]
    xx = [re.sub("대비", " 대비 " , a) for a in xx]
    xx = [re.sub("대용량", " 대용량 " , a) for a in xx]
    xx = [re.sub("대표", " 대표 " , a) for a in xx]
    xx = [re.sub("도래", " 도래 " , a) for a in xx]
    xx = [re.sub("독일", " 독일 " , a) for a in xx]
    xx = [re.sub("동의", " 동의 " , a) for a in xx]
    xx = [re.sub("등록", " 등록 " , a) for a in xx]
    xx = [re.sub("등록 일", " 등록일 " , a) for a in xx]
    xx = [re.sub("디지털", " 디지털 " , a) for a in xx]
    xx = [re.sub("뚜껑", " 뚜껑 " , a) for a in xx]
    xx = [re.sub("렌탈", " 렌탈 " , a) for a in xx]
    xx = [re.sub("론칭", " 론칭 " , a) for a in xx]
    xx = [re.sub("루헨스", " 루헨스 " , a) for a in xx]
    xx = [re.sub("리모컨", " 리모컨 " , a) for a in xx]
    xx = [re.sub("마사지", " 마사지 " , a) for a in xx]
    xx = [re.sub("마스크", " 마스크 " , a) for a in xx]
    xx = [re.sub("마트", " 마트 " , a) for a in xx]
    xx = [re.sub("만료", " 만료 " , a) for a in xx]
    xx = [re.sub("맥북", " 맥북 " , a) for a in xx]
    xx = [re.sub("먼지", " 먼지 " , a) for a in xx]
    xx = [re.sub("메뉴", " 메뉴 " , a) for a in xx]
    xx = [re.sub("면제", " 면제 " , a) for a in xx]
    xx = [re.sub("명의", " 명의 " , a) for a in xx]
    xx = [re.sub("명품", " 명품 " , a) for a in xx]
    xx = [re.sub("모델", " 모델 " , a) for a in xx]
    xx = [re.sub("모뎀", " 모뎀 " , a) for a in xx]
    xx = [re.sub("모바일", " 모바일 " , a) for a in xx]
    xx = [re.sub("목돈", " 목돈 " , a) for a in xx]
    xx = [re.sub("무게", " 무게 " , a) for a in xx]
    xx = [re.sub("무료", " 무료 " , a) for a in xx]
    xx = [re.sub("무비", " 무비 " , a) for a in xx]
    xx = [re.sub("무선", " 무선 " , a) for a in xx]
    xx = [re.sub("무이자", " 무이자 " , a) for a in xx]
    xx = [re.sub("문의", " 문의 " , a) for a in xx]
    xx = [re.sub("문자", " 문자 " , a) for a in xx]
    xx = [re.sub("물질", " 물질 " , a) for a in xx]
    xx = [re.sub("미납", " 미납 " , a) for a in xx]
    xx = [re.sub("미드나잇", " 미드나잇 " , a) for a in xx]
    xx = [re.sub("미세", " 미세 " , a) for a in xx]
    xx = [re.sub("바로", " 바로 " , a) for a in xx]
    xx = [re.sub("바이러스", " 바이러스 " , a) for a in xx]
    xx = [re.sub("박테리아", " 박테리아 " , a) for a in xx]
    xx = [re.sub("반값", " 반값 " , a) for a in xx]
    xx = [re.sub("반납", " 반납 " , a) for a in xx]
    xx = [re.sub("반환", " 반환 " , a) for a in xx]
    xx = [re.sub("발급", " 발급 " , a) for a in xx]
    xx = [re.sub("발생", " 발생 " , a) for a in xx]
    xx = [re.sub("발송", " 발송 " , a) for a in xx]
    xx = [re.sub("발행", " 발행 " , a) for a in xx]
    xx = [re.sub("방문", " 방문 " , a) for a in xx]
    xx = [re.sub("방법", " 방법 " , a) for a in xx]
    xx = [re.sub("방송", " 방송 " , a) for a in xx]
    xx = [re.sub("버튼", " 버튼 " , a) for a in xx]
    xx = [re.sub("번호", " 번호 " , a) for a in xx]
    xx = [re.sub("법률", " 법률 " , a) for a in xx]
    xx = [re.sub("베스트가전", " 베스트가전 " , a) for a in xx]
    xx = [re.sub("벽걸이", " 벽걸이 " , a) for a in xx]
    xx = [re.sub("변경", " 변경 " , a) for a in xx]
    xx = [re.sub("변상", " 변상 " , a) for a in xx]
    xx = [re.sub("별도", " 별도 " , a) for a in xx]
    xx = [re.sub("보급", " 보급 " , a) for a in xx]
    xx = [re.sub("보통", " 보통 " , a) for a in xx]
    xx = [re.sub("보호", " 보호 " , a) for a in xx]
    xx = [re.sub("본인", " 본인 " , a) for a in xx]
    xx = [re.sub("부가", " 부가 " , a) for a in xx]
    xx = [re.sub("부담", " 부담 " , a) for a in xx]
    xx = [re.sub("부산", " 부산 " , a) for a in xx]
    xx = [re.sub("부재", " 부재 " , a) for a in xx]
    xx = [re.sub("부전", " 부전 " , a) for a in xx]
    xx = [re.sub("부탁", " 부탁 " , a) for a in xx]
    xx = [re.sub("분납", " 분납 " , a) for a in xx]
    xx = [re.sub("분무기", " 분무기 " , a) for a in xx]
    xx = [re.sub("불가", " 불가 " , a) for a in xx]
    xx = [re.sub("불편", " 불편 " , a) for a in xx]
    xx = [re.sub("비키", " 비키 " , a) for a in xx]
    xx = [re.sub("빨래", " 빨래 " , a) for a in xx]
    xx = [re.sub("사용", " 사용 " , a) for a in xx]
    xx = [re.sub("사은품", " 사은품 " , a) for a in xx]
    xx = [re.sub("사전", " 사전 " , a) for a in xx]
    xx = [re.sub("사항", " 사항 " , a) for a in xx]
    xx = [re.sub("삼성", " 삼성 " , a) for a in xx]
    xx = [re.sub("상담", " 상담 " , a) for a in xx]
    xx = [re.sub("상담 원", " 상담원 " , a) for a in xx]
    xx = [re.sub("상당", " 상당 " , a) for a in xx]
    xx = [re.sub("상세", " 상세 " , a) for a in xx]
    xx = [re.sub("상품", " 상품 " , a) for a in xx]
    xx = [re.sub("새학기", " 새학기 " , a) for a in xx]
    xx = [re.sub("새해", " 새해 " , a) for a in xx]
    xx = [re.sub("생활필수품", " 생활필수품 " , a) for a in xx]
    xx = [re.sub("서비스", " 서비스 " , a) for a in xx]
    xx = [re.sub("서울", " 서울 " , a) for a in xx]
    xx = [re.sub("선명", " 선명 " , a) for a in xx]
    xx = [re.sub("선물", " 선물 " , a) for a in xx]
    xx = [re.sub("설정", " 설정 " , a) for a in xx]
    xx = [re.sub("설치", " 설치 " , a) for a in xx]
    xx = [re.sub("성인", " 성인 " , a) for a in xx]
    xx = [re.sub("성형", " 성형 " , a) for a in xx]
    xx = [re.sub("세탁", " 세탁 " , a) for a in xx]
    xx = [re.sub("센터", " 센터 " , a) for a in xx]
    xx = [re.sub("센터", " 센터 " , a) for a in xx]
    xx = [re.sub("소규모", " 소규모 " , a) for a in xx]
    xx = [re.sub("수신", " 수신 " , a) for a in xx]
    xx = [re.sub("스마트", " 스마트 " , a) for a in xx]
    xx = [re.sub("스 마트", " 스마트 " , a) for a in xx]
    xx = [re.sub("스크린", " 스크린 " , a) for a in xx]
    xx = [re.sub("스태프", " 스태프 " , a) for a in xx]
    xx = [re.sub("승인", " 승인 " , a) for a in xx]
    xx = [re.sub("시리즈", " 시리즈 " , a) for a in xx]
    xx = [re.sub("시스템", " 시스템 " , a) for a in xx]
    xx = [re.sub("시청", " 시청 " , a) for a in xx]
    xx = [re.sub("시행", " 시행 " , a) for a in xx]
    xx = [re.sub("신규", " 신규 " , a) for a in xx]
    xx = [re.sub("신상품", " 신상품 " , a) for a in xx]
    xx = [re.sub("신용", " 신용 " , a) for a in xx]
    xx = [re.sub("신청", " 신청 " , a) for a in xx]
    xx = [re.sub("신학기", " 신학기 " , a) for a in xx]
    xx = [re.sub("신형", " 신형 " , a) for a in xx]
    xx = [re.sub("실내", " 실내 " , a) for a in xx]
    xx = [re.sub("실속", " 실속 " , a) for a in xx]
    xx = [re.sub("아이패드", " 아이패드 " , a) for a in xx]
    xx = [re.sub("안내", " 안내 " , a) for a in xx]
    xx = [re.sub("안마", " 안마 " , a) for a in xx]
    xx = [re.sub("알뜰", " 알뜰 " , a) for a in xx]
    xx = [re.sub("약정", " 약정 " , a) for a in xx]
    xx = [re.sub("어르신", " 어르신 " , a) for a in xx]
    xx = [re.sub("어린아이", " 어린아이 " , a) for a in xx]
    xx = [re.sub("엄청", " 엄청 " , a) for a in xx]
    
    xx = [re.sub("업그레이드", " 업그레이드 " , a) for a in xx]
    xx = [re.sub("여부", " 여부 " , a) for a in xx]
    xx = [re.sub("연락", " 연락 " , a) for a in xx]
    xx = [re.sub("영동", " 영동 " , a) for a in xx]
    xx = [re.sub("영업", " 영업 " , a) for a in xx]
    xx = [re.sub("영화", " 영화 " , a) for a in xx]
    xx = [re.sub("예약", " 예약 " , a) for a in xx]
    xx = [re.sub("예정", " 예정 " , a) for a in xx]
    xx = [re.sub("오염", " 오염 " , a) for a in xx]
    xx = [re.sub("완료", " 완료 " , a) for a in xx]
    xx = [re.sub("요금", " 요금 " , a) for a in xx]
    xx = [re.sub("요청", " 요청 " , a) for a in xx]
    xx = [re.sub("용도", " 용도 " , a) for a in xx]
    xx = [re.sub("용량", " 용량 " , a) for a in xx]
    xx = [re.sub("위탁", " 위탁 " , a) for a in xx]
    xx = [re.sub("유료", " 유료 " , a) for a in xx]
    xx = [re.sub("유선", " 유선 " , a) for a in xx]
    xx = [re.sub("유통", " 유통 " , a) for a in xx]
    xx = [re.sub("유해", " 유해 " , a) for a in xx]
    xx = [re.sub("유효", " 유효 " , a) for a in xx]
    xx = [re.sub("은행", " 은행 " , a) for a in xx]
    xx = [re.sub("의무", " 의무 " , a) for a in xx]
    xx = [re.sub("이번", " 이번 " , a) for a in xx]
    xx = [re.sub("이번 달", " 이번달 " , a) for a in xx]
    xx = [re.sub("이상", " 이상 " , a) for a in xx]
    xx = [re.sub("이용", " 이용 " , a) for a in xx]
    xx = [re.sub("이체", " 이체 " , a) for a in xx]
    xx = [re.sub("인터넷", " 인터넷 " , a) for a in xx]
    xx = [re.sub("인터페이스", " 인터페이스 " , a) for a in xx]
    xx = [re.sub("인하", " 인하 " , a) for a in xx]
    xx = [re.sub("일반", " 일반 " , a) for a in xx]
    xx = [re.sub("입금", " 입금 " , a) for a in xx]
    xx = [re.sub("입력", " 입력 " , a) for a in xx]
    xx = [re.sub("입학", " 입학 " , a) for a in xx]
    xx = [re.sub("자녀", " 자녀 " , a) for a in xx]
    xx = [re.sub("잔액", " 잔액 " , a) for a in xx]
    xx = [re.sub("장비", " 장비 " , a) for a in xx]
    xx = [re.sub("장점", " 장점 " , a) for a in xx]
    xx = [re.sub("장착", " 장착 " , a) for a in xx]
    xx = [re.sub("적립", " 적립 " , a) for a in xx]
    xx = [re.sub("적용", " 적용 " , a) for a in xx]
    xx = [re.sub("적용", " 적용 " , a) for a in xx]
    xx = [re.sub("전기", " 전기 " , a) for a in xx]
    xx = [re.sub("전용", " 전용 " , a) for a in xx]
    xx = [re.sub("전입", " 전입 " , a) for a in xx]
    xx = [re.sub("전체", " 전체 " , a) for a in xx]
    xx = [re.sub("전화", " 전화 " , a) for a in xx]
    xx = [re.sub("접수", " 접수 " , a) for a in xx]
    xx = [re.sub("정기", " 정기 " , a) for a in xx]
    xx = [re.sub("정보", " 정보 " , a) for a in xx]
    xx = [re.sub("정상", " 정상 " , a) for a in xx]
    xx = [re.sub("제공", " 제공 " , a) for a in xx]
    xx = [re.sub("제조", " 제조 " , a) for a in xx]
    xx = [re.sub("제한", " 제한 " , a) for a in xx]
    xx = [re.sub("제한", " 제한 " , a) for a in xx]
    xx = [re.sub("조회", " 조회 " , a) for a in xx]
    xx = [re.sub("졸업", " 졸업 " , a) for a in xx]
    xx = [re.sub("종료", " 종료 " , a) for a in xx]
    xx = [re.sub("죄송", " 죄송 " , a) for a in xx]
    xx = [re.sub("주말", " 주말 " , a) for a in xx]
    xx = [re.sub("주식회사", " 주식회사 " , a) for a in xx]
    xx = [re.sub("줄거리", " 줄거리 " , a) for a in xx]
    xx = [re.sub("중지", " 중지 " , a) for a in xx]
    xx = [re.sub("증정", " 증정 " , a) for a in xx]
    xx = [re.sub("증정", " 증정 " , a) for a in xx]
    xx = [re.sub("지금", " 지금 " , a) for a in xx]
    xx = [re.sub("지급", " 지급 " , a) for a in xx]
    xx = [re.sub("지상파", " 지상파 " , a) for a in xx]
    xx = [re.sub("지연", " 지연 " , a) for a in xx]
    xx = [re.sub("직통", " 직통 " , a) for a in xx]
    xx = [re.sub("진행", " 진행 " , a) for a in xx]
    xx = [re.sub("진행", " 진행 " , a) for a in xx]
    xx = [re.sub("질문", " 질문 " , a) for a in xx]
    xx = [re.sub("질환", " 질환 " , a) for a in xx]
    xx = [re.sub("차단", " 차단 " , a) for a in xx]
    xx = [re.sub("차량 용", " 차량용 " , a) for a in xx]
    xx = [re.sub("차량용", " 차량용 " , a) for a in xx]
    xx = [re.sub("채널", " 채널 " , a) for a in xx]
    xx = [re.sub("처리", " 처리 " , a) for a in xx]
    xx = [re.sub("처음", " 처음 " , a) for a in xx]
    xx = [re.sub("청구", " 청구 " , a) for a in xx]
    xx = [re.sub("청구 서", " 청구서 " , a) for a in xx]
    xx = [re.sub("청 정기", " 청정기 " , a) for a in xx]
    xx = [re.sub("초과", " 초과 " , a) for a in xx]
    xx = [re.sub("촉진", " 촉진 " , a) for a in xx]
    xx = [re.sub("최고", " 최고 " , a) for a in xx]
    xx = [re.sub("최대", " 최대 " , a) for a in xx]
    xx = [re.sub("최신", " 최신 " , a) for a in xx]
    xx = [re.sub("최종", " 최종 " , a) for a in xx]
    xx = [re.sub("추가", " 추가 " , a) for a in xx]
    xx = [re.sub("축적", " 축적 " , a) for a in xx]
    xx = [re.sub("출금", " 출금 " , a) for a in xx]
    xx = [re.sub("출시", " 출시 " , a) for a in xx]
    xx = [re.sub("충전", " 충전 " , a) for a in xx]
    xx = [re.sub("추천", " 추천 " , a) for a in xx]    
    xx = [re.sub("취급", " 취급 " , a) for a in xx]
    xx = [re.sub("침투", " 침투 " , a) for a in xx]
    xx = [re.sub("카드", " 카드 " , a) for a in xx]
    xx = [re.sub("카메라", " 카메라 " , a) for a in xx]
    xx = [re.sub("캐논", " 캐논 " , a) for a in xx]
    xx = [re.sub("캐치온", " 캐치온 " , a) for a in xx]
    xx = [re.sub("캠페인", " 캠페인 " , a) for a in xx]
    xx = [re.sub("캔디", " 캔디 " , a) for a in xx]
    xx = [re.sub("컨텐츠", " 컨텐츠 " , a) for a in xx]
    xx = [re.sub("코드제로", " 코드제로 " , a) for a in xx]
    xx = [re.sub("코인", " 코인 " , a) for a in xx]
    xx = [re.sub("쿠폰", " 쿠폰 " , a) for a in xx]
    xx = [re.sub("키즈", " 키즈 " , a) for a in xx]
    xx = [re.sub("탑재", " 탑재 " , a) for a in xx]
    xx = [re.sub("태블릿", " 태블릿 " , a) for a in xx]
    xx = [re.sub("터치", " 터치 " , a) for a in xx]
    xx = [re.sub("통돌이", " 통돌이 " , a) for a in xx]
    xx = [re.sub("통신", " 통신 " , a) for a in xx] 
    xx = [re.sub("통화", " 통화 " , a) for a in xx]
    xx = [re.sub("투하", " 투하 " , a) for a in xx]
    xx = [re.sub("트롬", " 트롬 " , a) for a in xx]
    xx = [re.sub("특가", " 특가 " , a) for a in xx]
    xx = [re.sub("파격", " 파격 " , a) for a in xx]
    xx = [re.sub("판매", " 판매 " , a) for a in xx]
    xx = [re.sub("페이백", " 페이백 " , a) for a in xx]
    xx = [re.sub("페이지", " 페이지 " , a) for a in xx]
    xx = [re.sub("포함", " 포함 " , a) for a in xx]
    xx = [re.sub("품절", " 품절 " , a) for a in xx]
    xx = [re.sub("프로모션", " 프로모션 " , a) for a in xx]
    xx = [re.sub("프로젝터", " 프로젝터 " , a) for a in xx]
    xx = [re.sub("프리미엄", " 프리미엄 " , a) for a in xx]
    xx = [re.sub("필수", " 필수 " , a) for a in xx]
    xx = [re.sub("필요", " 필요 " , a) for a in xx]
    xx = [re.sub("필터", " 필터 " , a) for a in xx]
    xx = [re.sub("하루", " 하루 " , a) for a in xx]
    xx = [re.sub("한정", " 한정 " , a) for a in xx]
    xx = [re.sub("한파", " 한파 " , a) for a in xx]
    xx = [re.sub("할부", " 할부 " , a) for a in xx]
    xx = [re.sub("할인", " 할인 " , a) for a in xx]
    xx = [re.sub("항목", " 항목 " , a) for a in xx]
    xx = [re.sub("해결", " 해결 " , a) for a in xx]
    xx = [re.sub("해지", " 해지 " , a) for a in xx]
    xx = [re.sub("헬로", " 헬로 " , a) for a in xx]
    xx = [re.sub("현금", " 현금 " , a) for a in xx]
    xx = [re.sub("혜택", " 혜택 " , a) for a in xx]
    xx = [re.sub("혜택", " 혜택 " , a) for a in xx]
    xx = [re.sub("호흡", " 호흡 " , a) for a in xx]
    xx = [re.sub("홈페이지", " 홈페이지 " , a) for a in xx]
    xx = [re.sub("화면", " 화면 " , a) for a in xx]
    xx = [re.sub("화질", " 화질 " , a) for a in xx]
    xx = [re.sub("확인", " 확인 " , a) for a in xx]
    xx = [re.sub("확정", " 확정 " , a) for a in xx]
    xx = [re.sub("환불", " 환불 " , a) for a in xx]
    xx = [re.sub("회사", " 회사 " , a) for a in xx]
    xx = [re.sub("희망", " 희망 " , a) for a in xx]
    xx = [re.sub("희망 일", " 희망일 " , a) for a in xx]
    xx = [re.sub("가입 자", " 가입자 " , a) for a in xx]
    xx = [re.sub("건조 기", " 건조기 " , a) for a in xx]
    xx = [re.sub("건조기", " 건조기 " , a) for a in xx]
    xx = [re.sub("고객 명", " 고객명 " , a) for a in xx]
    xx = [re.sub("공기 청정기", " 공기청정기 " , a) for a in xx]
    xx = [re.sub("관련서류", " 관련서류 " , a) for a in xx]
    xx = [re.sub("기타  금액", " 기타금액 " , a) for a in xx]
    xx = [re.sub("기타 금액", " 기타금액 " , a) for a in xx]
    xx = [re.sub("김치 냉장고", " 김치냉장고 " , a) for a in xx]
    xx = [re.sub("김치냉장고", " 김치냉장고 " , a) for a in xx]
    xx = [re.sub("나의 TV", " 나의TV " , a) for a in xx]
    xx = [re.sub("납금액", " 납금액 " , a) for a in xx]
    xx = [re.sub("냉온 정수기", " 냉온정수기 " , a) for a in xx]
    xx = [re.sub("냉온정수기", " 냉온정수기 " , a) for a in xx]
    xx = [re.sub("다리 마사지", " 다리마사지 " , a) for a in xx]
    xx = [re.sub("다리마사지", " 다리마사지 " , a) for a in xx]
    xx = [re.sub("다시 보기", " 다시보기 " , a) for a in xx]
    xx = [re.sub("단체  가입", " 단체가입 " , a) for a in xx]
    xx = [re.sub("단체 가입", " 단체가입 " , a) for a in xx]
    xx = [re.sub("마이 컨텐츠", " 마이컨텐츠 " , a) for a in xx]
    xx = [re.sub("마이컨텐츠", " 마이컨텐츠 " , a) for a in xx]
    xx = [re.sub("만원 상당", " 만원상당 " , a) for a in xx]
    xx = [re.sub("메시지", " 메시지 " , a) for a in xx]
    xx = [re.sub("메시지 함", " 메세지함 " , a) for a in xx]
    xx = [re.sub("미납 금", " 미납금 " , a) for a in xx]
    xx = [re.sub("바로 가기", " 바로가기 " , a) for a in xx]
    xx = [re.sub("반환 금", " 반환금 " , a) for a in xx]
    xx = [re.sub("부가 세", " 부가세 " , a) for a in xx]
    xx = [re.sub("비밀 번호", " 비밀번호 " , a) for a in xx]
    xx = [re.sub("세탁 기", " 세탁기 " , a) for a in xx]
    xx = [re.sub("스마트 폰", " 스마트폰 " , a) for a in xx]
    xx = [re.sub("영업 팀", " 영업팀 " , a) for a in xx]
    xx = [re.sub("예정 일", " 예정일 " , a) for a in xx]
    xx = [re.sub("월별 요금", " 월별요금 " , a) for a in xx]
    xx = [re.sub("월별요금", " 월별요금 " , a) for a in xx]
    xx = [re.sub("월사용 금액", " 월사용금액 " , a) for a in xx]
    xx = [re.sub("월사용금액", " 월사용금액 " , a) for a in xx]
    xx = [re.sub("월정액", " 월정액 " , a) for a in xx]
    xx = [re.sub("월정액 가입", " 월정액가입 " , a) for a in xx]
    xx = [re.sub("의류 건조기", " 의류건조기 " , a) for a in xx]
    xx = [re.sub("의류건조기", " 의류건조기 " , a) for a in xx]
    xx = [re.sub("이용  금액", " 이용금액 " , a) for a in xx]
    xx = [re.sub("이용 권", " 이용권 " , a) for a in xx]
    xx = [re.sub("이용 금액", " 이용금액 " , a) for a in xx]
    xx = [re.sub("입력 하기", " 입력하기 " , a) for a in xx]
    xx = [re.sub("자동 이체", " 자동이체 " , a) for a in xx]
    xx = [re.sub("자동 할인", " 자동할인 " , a) for a in xx]
    xx = [re.sub("재 발송", " 재발송 " , a) for a in xx]
    xx = [re.sub("재 확인", " 재확인 " , a) for a in xx]
    xx = [re.sub("재발송", " 재발송 " , a) for a in xx]
    xx = [re.sub("재확인", " 재확인 " , a) for a in xx]
    xx = [re.sub("전기  요금", " 전기요금 " , a) for a in xx]
    xx = [re.sub("제휴 서비스", " 제휴서비스 " , a) for a in xx]
    xx = [re.sub("중도", " 중도 " , a) for a in xx]
    xx = [re.sub("중도  해지", " 중도해지 " , a) for a in xx]
    xx = [re.sub("중도 해지", " 중도해지 " , a) for a in xx]
    xx = [re.sub("처리 일자", " 처리일자 " , a) for a in xx]
    xx = [re.sub("청소", " 청소 " , a) for a in xx]
    xx = [re.sub("청소 기", " 청소기 " , a) for a in xx]
    xx = [re.sub("청정 기", " 청정기 " , a) for a in xx]
    xx = [re.sub("청정기", " 청정기 " , a) for a in xx]
    xx = [re.sub("총 금액", " 총금액 " , a) for a in xx]
    xx = [re.sub("총금액", " 총금액 " , a) for a in xx]
    xx = [re.sub("최신 가전", " 최신가전 " , a) for a in xx]
    xx = [re.sub("최신 형", " 최신형 " , a) for a in xx]
    xx = [re.sub("추가  증정", " 추가증정 " , a) for a in xx]
    xx = [re.sub("추가 증정", " 추가증정 " , a) for a in xx]
    xx = [re.sub("충전  쿠폰", " 충전쿠폰 " , a) for a in xx]
    xx = [re.sub("케이블", " 케이블 " , a) for a in xx]
    xx = [re.sub("케이블  방송", " 케이블방송 " , a) for a in xx]
    xx = [re.sub("케이블 방송", " 케이블방송 " , a) for a in xx]
    xx = [re.sub("코인  증정", " 코인증정 " , a) for a in xx]
    xx = [re.sub("코인  충전", " 코인충전 " , a) for a in xx]
    xx = [re.sub("코인  충전  소", " 코인충전소 " , a) for a in xx]
    xx = [re.sub("코인  충전 소", " 코인충전소 " , a) for a in xx]
    xx = [re.sub("코인 증정", " 코인증정 " , a) for a in xx]
    xx = [re.sub("코인 충전", " 코인충전 " , a) for a in xx]
    xx = [re.sub("코인 충전  소", " 코인충전소 " , a) for a in xx]
    xx = [re.sub("코인 충전 소", " 코인충전소 " , a) for a in xx]
    xx = [re.sub("쿠폰  등록", " 쿠폰등록 " , a) for a in xx]
    xx = [re.sub("쿠폰  지급", " 쿠폰지급 " , a) for a in xx]
    xx = [re.sub("쿠폰 등록", " 쿠폰등록 " , a) for a in xx]
    xx = [re.sub("쿠폰 지급", " 쿠폰지급 " , a) for a in xx]
    xx = [re.sub("키즈 매니저", " 키즈매니저 " , a) for a in xx]
    xx = [re.sub("통화  버튼", " 통화버튼 " , a) for a in xx]
    xx = [re.sub("통화 버튼", " 통화버튼 " , a) for a in xx]
    xx = [re.sub("할인  구매", " 할인구매 " , a) for a in xx]
    xx = [re.sub("할인  반환", " 할인반환 " , a) for a in xx]
    xx = [re.sub("할인  반환  금", " 할인반환금 " , a) for a in xx]
    xx = [re.sub("할인  반환 금", " 할인반환금 " , a) for a in xx]
    xx = [re.sub("할인  이벤트", " 할인이벤트 " , a) for a in xx]
    xx = [re.sub("할인  적용", " 할인적용 " , a) for a in xx]
    xx = [re.sub("할인  정책", " 할인정책 " , a) for a in xx]
    xx = [re.sub("할인  쿠폰", " 할인쿠폰 " , a) for a in xx]
    xx = [re.sub("할인  판매", " 할인판매 " , a) for a in xx]
    xx = [re.sub("할인 가", " 할인가 " , a) for a in xx]
    xx = [re.sub("할인 구매", " 할인구매 " , a) for a in xx]
    xx = [re.sub("할인 구매", " 할인구매 " , a) for a in xx]
    xx = [re.sub("할인 반환", " 할인반환 " , a) for a in xx]
    xx = [re.sub("할인 반환", " 할인반환 " , a) for a in xx]
    xx = [re.sub("할인 반환 금", " 할인반환금 " , a) for a in xx]
    xx = [re.sub("할인 반환 금", " 할인반환금 " , a) for a in xx]
    xx = [re.sub("할인 이벤트", " 할인이벤트 " , a) for a in xx]
    xx = [re.sub("할인 적용", " 할인적용 " , a) for a in xx]
    xx = [re.sub("할인 정책", " 할인정책 " , a) for a in xx]
    xx = [re.sub("할인 쿠폰", " 할인쿠폰 " , a) for a in xx]
    xx = [re.sub("할인 판매", " 할인판매 " , a) for a in xx]
    xx = [re.sub("해지  요청", " 해지요청 " , a) for a in xx]
    xx = [re.sub("해지  희망", " 해지희망 " , a) for a in xx]
    xx = [re.sub("해지 요청", " 해지요청 " , a) for a in xx]
    xx = [re.sub("해지 희망", " 해지희망 " , a) for a in xx]
    xx = [re.sub("헬로  렌탈", " 헬로렌탈 " , a) for a in xx]
    xx = [re.sub("헬로  모바일", " 헬로모바일 " , a) for a in xx]
    xx = [re.sub("헬로 TV", " 헬로TV " , a) for a in xx]
    xx = [re.sub("헬로 모바일", " 헬로모바일 " , a) for a in xx]
    xx = [re.sub("헬로 폰", " 헬로폰 " , a) for a in xx]
    xx = [re.sub("헬로폰", " 헬로폰 " , a) for a in xx]
    return xx
"""
  xx = [re.sub("(광고)", " ", a) for a in xx]
  xx = [re.sub("납부안 내", "납부 안내", a) for a in xx]
  xx = [re.sub("입니다", " ", a) for a in xx]
  xx = [re.sub("으로", " ", a) for a in xx]
  xx = [re.sub("해지되지", " 해지 ", a) for a in xx]
  xx = [re.sub("cj헬로tv", " cj헬로tv ", a) for a in xx]
  xx = [re.sub("부천방송", " 부천방송 ", a) for a in xx]
  xx = [re.sub("나라방송", " 나라방송 ", a) for a in xx]
  xx = [re.sub("위약금삭제요청등록되었습니다", " 위약금 삭제 요청 등록 ", a) for a in xx]
  xx = [re.sub("연락", " 연락 ", a) for a in xx]
  xx = [re.sub("확인", " 확인 ", a) for a in xx]
  xx = [re.sub("담당", " 담당 ", a) for a in xx]
  xx = [re.sub("통화", " 통화 ", a) for a in xx]
  xx = [re.sub("연결", " 연결 ", a) for a in xx]
  xx = [re.sub("문의", " 문의 ", a) for a in xx]
  xx = [re.sub("명의", " 명의 ", a) for a in xx]
  xx = [re.sub("미납금", " 미납금 ", a) for a in xx]
  xx = [re.sub("해지", " 해지 ", a) for a in xx]
  xx = [re.sub("전화", " 전화 ", a) for a in xx]
  xx = [re.sub("시간", " 시간 ", a) for a in xx]
  xx = [re.sub("부재", " 부재 ", a) for a in xx]
  xx = [re.sub("회수", " 회수 ", a) for a in xx]
  xx = [re.sub("방문", " 방문 ", a) for a in xx]
  xx = [re.sub("발송", " 발송 ", a) for a in xx]
  xx = [re.sub("접수", " 접수 ", a) for a in xx]
  xx = [re.sub("취소", " 취소 ", a) for a in xx]
  xx = [re.sub("주소", " 주소 ", a) for a in xx]
  xx = [re.sub("설치", " 설치 ", a) for a in xx]
  xx = [re.sub("사용", " 사용 ", a) for a in xx]
  xx = [re.sub("면제", " 면제 ", a) for a in xx]
  xx = [re.sub("소요", " 소요 ", a) for a in xx]
  xx = [re.sub("계획", " 계획 ", a) for a in xx]
  xx = [re.sub("완료", " 완료 ", a) for a in xx]
  xx = [re.sub("번호", " 번호 ", a) for a in xx]
  xx = [re.sub("가입", " 가입 ", a) for a in xx]
  xx = [re.sub("요금", " 요금 ", a) for a in xx]
  xx = [re.sub("불편", " 불편 ", a) for a in xx]
  xx = [re.sub("정상청구", " 정상 청구 ", a) for a in xx]
  xx = [re.sub("보류", " 보류 ", a) for a in xx]
  xx = [re.sub("무료수신거부", " 무료수신거부 ", a) for a in xx]
  xx = [re.sub("저장", " 저장 ", a) for a in xx]
  xx = [re.sub("이사", " 이사 ", a) for a in xx]
  xx = [re.sub("전달", " 전달 ", a) for a in xx]
  xx = [re.sub("정보", " 정보 ", a) for a in xx]
  xx = [re.sub("아이디", " 아이디 ", a) for a in xx]
  xx = [re.sub("상담원", " 상담원 ", a) for a in xx]
  
  
 """   
  
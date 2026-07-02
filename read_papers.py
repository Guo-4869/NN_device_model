import pdfplumber, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ref = r'D:\gzp\研究生\半导体器件建模与仿真\references\files'
papers = {
    '1903': 'SPICE-Compatible NN Compact Model (Tung 2024)',
    '1913': 'BSIM-NN Framework (Tung & Hu 2023)',
    '2089': 'BSIM-NN ML (Tung 2025)',
    '1923': 'NN Self-Heating (Tung 2024)',
    '1924': 'NN NQS Model (Tung 2024)',
    '1643': 'NN MOSFET Compact (Wei 2023)',
    '1685': 'Physics-Based NN MOS (Huang 2023)',
    '1810': 'Deep Learning MOS IV (Kao 2022)',
    '1591': 'ResNet Device Modeling (Bavi 2024)',
    '1502': 'ML Device Modeling Survey (Zhang 2024)',
    '1593': 'Emerging Device Models (Li 2024)',
    '1499': 'AI/ML SPICE Benchmarks (McAndrew 2025)',
}
sep = "=" * 60
for folder, title in papers.items():
    try:
        dirpath = os.path.join(ref, folder)
        files = os.listdir(dirpath)
        pdfs = [f for f in files if f.endswith('.pdf')]
        if not pdfs:
            continue
        path = os.path.join(dirpath, pdfs[0])
        pdf = pdfplumber.open(path)
        print("\n" + sep)
        print("[%s] %s" % (folder, title))
        print("Pages: %d" % len(pdf.pages))
        print(sep)
        for i in range(min(3, len(pdf.pages))):
            t = pdf.pages[i].extract_text()
            if t:
                t = t.encode('ascii', 'replace').decode('ascii')
                print("--- Page %d ---" % (i+1))
                print(t[:2000])
        pdf.close()
    except Exception as e:
        print("[%s] ERROR: %s" % (folder, str(e)))
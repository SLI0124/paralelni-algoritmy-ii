# Paralelní algoritmy II

Cílem pøedmìtu je doplnit a rozšíøit témata, se kterımi se mohl posluchaè seznámit v pøedmìtu Paralelní algoritmy I (PAI). Vzhledem k charakteru a úèelu pøedmìtu se bude jednat o témata, která budou ilustrovat vyuití grafickıch procesorù (GPU) pro øešení algoritmickıch úloh. Studenti budou blíe seznámeni s existujícími architekturami GPU a frameworky pro paralelní programování. S ohledem na skuteènost, e na VŠB vzniklo centrum nVidia Research, bude blíe vysvìtlována architektura nVidia CUDA. Jedním z cílù je pøedat posluchaèùm znalosti, které mohou vyuít pøi øešení praktickıch úloh a u v rámci magisterskıch prací èi grantovıch projektù realizovanıch na VŠB. 
Získané znalosti a vìdomosti: 
- orientace v základní architektuøe grafickıch procesorù (GPU) 
- znalost softwarové architektury paralelního programu, štìpení úlohy do gridù, blokù, vláken 
- znalost vybraného frameworku pro paralelní programování na GPU 
- pochopení problematiky algoritmizace, pøevod sériovıch úloh na paralelní 
- posouzení distribuce paralelní úlohy na více GPU, clusterù 
- zvládnutí implementace praktické úlohy zpracování dat

Odkaz na stránky pøedmìtu: [Paralelní algoritmy II](http://gajdos.cs.vsb.cz/en/parallel-applications-2).

## Instalace

Pro instalaci je potøeba mít nainstalovanı CUDA Toolkit verze 12.6. a vyšší. 

Postupoval jsem to i podruhé a vše fungovalo. Vìtšinou jde o chybìjící .exe a .dll soubory, které se nanahrávají na GitHub.

### Naètení projektu do Visual Studio

Toto by mìlo bıt u nastaveno, ale pro jistotu to píšu znova, pokud by nìkdo narazil na tento problém.

1. Odpojte projekt (Unload project).  
2. Dvakrát kliknìte na název projektu ve stromu projektù.  
3. Aktuálnì jsou dvì místa, kde je specifikována verze CUDA Toolkit, zmìòte ji na verzi, kterou máte nainstalovanou:  
   - na øádku, kterı øíká `    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 12.8.props" />`  
   - a na øádku `    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 12.8.props" />`  
4. Ulote vlastnosti projektu (otevøenı upravenı soubor).  
5. Znovu naètìte projekt.  

#### Staení DLL a EXE souborù

#### FreeGLUT

Z této stránky si stáhnìte FreeGLUT binárky pro Windows, sekce MSVC Package: [FreeGLUT](https://www.transmissionzero.co.uk/software/freeglut-devel/).

Vytvoøte následující cestu: `common/FreeGLUT/windows/bin/x64/` a do ní zkopírujte sloku `x64` z FreeGLUT binárek. Sloka by mìla nyní obsahovat soubory:
- freeglut.dll

#### FreeImage

Z této stránky si stáhnìte FreeImage binárky pro Windows, sekce MSVC Package: [FreeImage](https://freeimage.sourceforge.io/download.html).

Vytvoøte následující cestu: `common/FreeImage/windows/bin/x64/` a do ní zkopírujte sloku `x64` z FreeImage binárek. Sloka by mìla nyní obsahovat soubory:
- FreeImage.dll

#### GLEW

Z této stránky si stáhnìte GLEW binárky pro Windows: [GLEW](https://glew.sourceforge.net/).
Vytvoøte následující cestu: `common/Glew/windows/bin/x64/` a do ní zkopírujte sloku `x64` z GLEW binárek. Sloka by mìla nyní obsahovat soubory:
- glew64.dll (glew32.dll pro 32bitovou verzi by mìl také fungovat, kdytak staèí pøejmenovat, mìlo by to fungovat jak pro 32bitovou, tak 64bitovou verzi)
- glewinfo.exe
- visualinfo.exe

#### Windows 2015 Redist

Bìhem sestavení jsem narazil na problém s chybìjícími redistributable balíèky. Na stejnı problém narazili i [zde](https://answers.microsoft.com/en-us/windows/forum/all/missing-vcomp140dll/afca0b6b-3ced-4e82-8ce8-8734a440d516).
Proto je potøeba stáhnout a nainstalovat Visual C++ Redistributable for Visual Studio 2015, monost x64.

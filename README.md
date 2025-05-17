# Paraleln� algoritmy II

C�lem p�edm�tu je doplnit a roz���it t�mata, se kter�mi se mohl poslucha� sezn�mit v p�edm�tu Paraleln� algoritmy I (PAI). Vzhledem k charakteru a ��elu p�edm�tu se bude jednat o t�mata, kter� budou ilustrovat vyu�it� grafick�ch procesor� (GPU) pro �e�en� algoritmick�ch �loh. Studenti budou bl�e sezn�meni s existuj�c�mi architekturami GPU a frameworky pro paraleln� programov�n�. S ohledem na skute�nost, �e na V�B vzniklo centrum nVidia Research, bude bl�e vysv�tlov�na architektura nVidia CUDA. Jedn�m z c�l� je p�edat poslucha��m znalosti, kter� mohou vyu��t p�i �e�en� praktick�ch �loh a� u� v r�mci magistersk�ch prac� �i grantov�ch projekt� realizovan�ch na V�B. 
Z�skan� znalosti a v�domosti: 
- orientace v z�kladn� architektu�e grafick�ch procesor� (GPU) 
- znalost softwarov� architektury paraleln�ho programu, �t�pen� �lohy do grid�, blok�, vl�ken 
- znalost vybran�ho frameworku pro paraleln� programov�n� na GPU 
- pochopen� problematiky algoritmizace, p�evod s�riov�ch �loh na paraleln� 
- posouzen� distribuce paraleln� �lohy na v�ce GPU, cluster� 
- zvl�dnut� implementace praktick� �lohy zpracov�n� dat

Odkaz na str�nky p�edm�tu: [Paraleln� algoritmy II](http://gajdos.cs.vsb.cz/en/parallel-applications-2).

## Instalace

Pro instalaci je pot�eba m�t nainstalovan� CUDA Toolkit verze 12.6. a vy���. 

Postupoval jsem to i podruh� a v�e fungovalo. V�t�inou jde o chyb�j�c� .exe a .dll soubory, kter� se nanahr�vaj� na GitHub.

### Na�ten� projektu do Visual Studio

Toto by m�lo b�t u� nastaveno, ale pro jistotu to p�u znova, pokud by n�kdo narazil na tento probl�m.

1. Odpojte projekt (Unload project).  
2. Dvakr�t klikn�te na n�zev projektu ve stromu projekt�.  
3. Aktu�ln� jsou dv� m�sta, kde je specifikov�na verze CUDA Toolkit, zm��te ji na verzi, kterou m�te nainstalovanou:  
   - na ��dku, kter� ��k� `    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 12.8.props" />`  
   - a na ��dku `    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 12.8.props" />`  
4. Ulo�te vlastnosti projektu (otev�en� upraven� soubor).  
5. Znovu na�t�te projekt.  

#### Sta�en� DLL a EXE soubor�

#### FreeGLUT

Z t�to str�nky si st�hn�te FreeGLUT bin�rky pro Windows, sekce MSVC Package: [FreeGLUT](https://www.transmissionzero.co.uk/software/freeglut-devel/).

Vytvo�te n�sleduj�c� cestu: `common/FreeGLUT/windows/bin/x64/` a do n� zkop�rujte slo�ku `x64` z FreeGLUT bin�rek. Slo�ka by m�la nyn� obsahovat soubory:
- freeglut.dll

#### FreeImage

Z t�to str�nky si st�hn�te FreeImage bin�rky pro Windows, sekce MSVC Package: [FreeImage](https://freeimage.sourceforge.io/download.html).

Vytvo�te n�sleduj�c� cestu: `common/FreeImage/windows/bin/x64/` a do n� zkop�rujte slo�ku `x64` z FreeImage bin�rek. Slo�ka by m�la nyn� obsahovat soubory:
- FreeImage.dll

#### GLEW

Z t�to str�nky si st�hn�te GLEW bin�rky pro Windows: [GLEW](https://glew.sourceforge.net/).
Vytvo�te n�sleduj�c� cestu: `common/Glew/windows/bin/x64/` a do n� zkop�rujte slo�ku `x64` z GLEW bin�rek. Slo�ka by m�la nyn� obsahovat soubory:
- glew64.dll (glew32.dll pro 32bitovou verzi by m�l tak� fungovat, kdy�tak sta�� p�ejmenovat, m�lo by to fungovat jak pro 32bitovou, tak 64bitovou verzi)
- glewinfo.exe
- visualinfo.exe

#### Windows 2015 Redist

B�hem sestaven� jsem narazil na probl�m s chyb�j�c�mi redistributable bal��ky. Na stejn� probl�m narazili i [zde](https://answers.microsoft.com/en-us/windows/forum/all/missing-vcomp140dll/afca0b6b-3ced-4e82-8ce8-8734a440d516).
Proto je pot�eba st�hnout a nainstalovat Visual C++ Redistributable for Visual Studio 2015, mo�nost x64.

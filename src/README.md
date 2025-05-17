# Zadání jednotlivıch úkolù na cvièení

## Cvièení 1 - Seznámení a znalosti C++

- Alokujte HOST pamì, která bude reprezentovat dva M-dimenzionální vektory (A, B), a naplòte je nìjakımi hodnotami, kde M je dostateènì velké. Zaènìte však s malım M, abyste bìhem ladìní vidìli rozumné vıstupy.
- Alokujte DEVICE pamì, abyste mohli kopírovat data z HOST.
- Alokujte DEVICE pamì pro uloení vıstupního M-dimenzionálního vektoru C.
- Vytvoøte kernel, kterı sèítá skalární hodnoty tak, e $C_i = A_i + B_i$, kde $i\in <0,m-1>$.
- Alokujte HOST pamì, která bude reprezentovat 2 M-dimenzionální vektory $A=[a_0,\dots,a_{m-1}]$, $B=[b_0,\dots,b_{m-1}]$, a naplòte je nìjakımi hodnotami.
- Alokujte DEVICE pamì, abyste mohli kopírovat data z HOST.
- Alokujte DEVICE pamì pro uloení vıstupních M-dimenzionálních vektorù $C=[c_0,\dots, c_{m-1}]$.
- Vytvoøte kernel, kterı sèítá vektory tak, e $C_i = A_i + B_i$.
- ZAMYSLETE SE NAD VARIANTAMI VAŠEHO ØEŠENÍ, ZVATE KLADY A ZÁPORY.

## Cvièení 2 - alokace pamìti, page-locked pamì

- Vytvoøte sloupcovou matici $M[mRows\times mCols]$ obsahující èísla 0, 1, 2, 3 $\dots$
- Data by mìla bıt správnì zarovnána v pamìti uzamèené na stránce (page-locked memory).
- Matice by mìla bıt naplnìna v CUDA kernelu.
- Musíte pouít Pitch CUDA pamì s odpovídajícím zarovnáním.
- Pro zpracování dat musí bıt pouita 2D møíka 2D blokù o velikosti 8x8.
- Hodnoty matice inkrementujte.
- Nakonec zkopírujte matici na HOST pomocí funkce cudaMemcpy2D.

## Cvièení 3 - sdílená pamì

- Mìjme jednoduchı èásticovı systém reprezentující sadu pozic $N$ dešovıch kapek v 3D prostoru, kde $N \geq (1<<20)$.
- Vytvoøte vhodnou datovou reprezentaci zmínìné sady dešovıch kapek.
- Mìjme pole 256 vìtrnıch elektráren, které poskytují 256 pohybovıch vektorù. Tyto pohybové vektory zpùsobují zmìny pozic všech dešovıch kapek za jednu sekundu.
- Vytvoøte kernel, kterı simuluje pád dešovıch kapek.
- Pro zjednodušení pøedpokládejme, e jedno volání kernelu simuluje jednu sekundu v simulovaném svìtì.

## Cvièení 4 - konstantní pamì

- Zkuste napsat jednoduchı kód, kterı alokuje a nastaví skalární hodnotu v konstantní pamìti GPU.
- Zkopírujte data zpìt na HOST a zkontrolujte hodnotu.

- Udìlejte toté s vlastní strukturou a poté s nìjakım polem.

- Na hostiteli vytvoøte pole nazvané Reference s alespoò $2^{23}$ náhodnımi hodnotami s plovoucí desetinnou èárkou z intervalu $<0, 1>$.
- Vytvoøte pole nazvané Pattern s alespoò 16 náhodnımi hodnotami s plovoucí desetinnou èárkou z intervalu $<0, 1>$.
- Vytvoøte metodu, která najde nejlepší shodu Pattern v poli Reference. 
- Otázky, na které musíte odpovìdìt pøedem:
    - Jak a kde data uloit?
    - Jaká je funkce pro porovnání?
    - Jak najít nejlepší shodu?
    - Potøebuji nìjakou další datovou strukturu?
    - Jak vrátit vısledek?

## Cvièení 5 - texturová pamì

TODO

## Cvièení 6 - unifikovaná pamì a Texture Object APU

Naètìte vıškovou mapu a pøipravte texturu pomocí CUDA Texture Object API. Vytvoøte normálovou mapu z vstupního obrázku pomocí Sobelova operátoru.

## Cvièení 7 - Texture reference API vs Object API

OpenGL, freeglut, glew, textury, pixel buffer object

- Postupujte podle pokynù ze semináøe.
- Poté se pokuste vyøešit tyto úkoly napsané v souboru Runner.cu.

## Cvièení 8 - Page-locked pamìt a atomické operace

- Podívejte se na alokaci pamìti v zdrojovém kódu -> cudaHostAlloc -> pamì uzamèená na stránce na hostiteli.
- Najdìte maximální èíslo v poli s pouitím atomickıch instrukcí.
- Zamyslete se nad úpravami kódu s pouitím lokálního úloištì dat "per block".
- Zamyslete se nad úpravami kódu s pouitím lokálního úloištì dat ve sdílené pamìti.
- Zamyslete se nad optimalizacemi.
- Zmìòte kód tak, aby pamì uzamèená na stránce byla nahrazena unifikovanou pamìtí.
- Zkuste profilovat svùj kód v NSight Systems (domácí úkol).
- Proveïte detailní analızu kernelù v NSight Compute (domácí úkol).

## Cvièení 9 - CUDA Streams

- Pokuste se dokonèit danou aplikaci. K tomu musíte implementovat všechny dílèí úkoly v kódu. Existují dva vektory A a B ($dim \sim= 2^{20}$), které budou N-krát duplikovány v cyklu. Jednoduchı kernel provádí souèet vektorù A+B=C. Vše bude provedeno pomocí streamù s ohledem na následující úkoly.
    - ÚKOL 1: Jednoduchı stream
    - ÚKOL 2: Dva streamy - pøístup hloubky (depth first approach)
    - ÚKOL 3: Dva streamy - pøístup šíøky (breadth first approach)

## Cvièení 10 - CUBLAS knihovna

- Pokuste se dokonèit danou aplikaci. K tomu musíte implementovat všechny dílèí úkoly v kódu. Musíte vytvoøit matici vzdáleností, která bude obsahovat vzdálenosti mezi body v prostoru o dimenzi dim.
- Vzdálenosti mohou bıt vypoèítány nìkolika rùznımi zpùsoby, ale zkuste pouít funkce BLAS3. To znamená, e musíte pracovat s maticovımi operacemi.
- Na závìr zvyšte poèet bodù a dimenzi.

# Zadání jednotlivıch úkolù za kredit

## Credit task 1 - Zpracování objektù v n-dimenzionálním prostoru

Pøedstavte si, e se nacházíte v n-dimenzionálním prostoru, kde je kadı objekt popsán jako n-dimenzionální vektor reálnıch èísel z libovolného rozsahu, ne nutnì z intervalu <0.0; 1.0>. Vygenerujte dostateènı poèet tìchto objektù do jedné velké matice M0 [m × n], kde m > 2^20 a n je dostateènì velké vzhledem k dostupné pamìti. Zamyslete se nad orientací matice (øádková/sloupcová).

1. Diskretizujte zadanou matici podle pravidel, která si urèíte, tak aby byl kadı objekt/vektor nakonec reprezentován vektorem prvkù typu uint8_t nebo uint32_t. Pravidla si mùete nastavit libovolnì, ale všechna nastavení musí bıt specifikována ve formì parametrù metody. Musí bıt zachována topologie dat – tzn. normalizace a další operace musí bıt provedeny ve stejném prostoru a všechny vektory musí bıt pøevedeny shodnì. Vıstupem bude nová matice M1 se stejnou orientací jako M0.
2. Najdìte objekt, kterı je nejvzdálenìjší od poèátku nového vektorového prostoru (nulového vektoru).
3. Najdìte prvek, kterı je nejvzdálenìjší od právì nalezeného objektu.
4. Pomocí našich pomocnıch rutin experimentálnì ovìøte dobu bìhu kadého kernelu.

Maximalizujte poèet operací, které mohou probíhat paralelnì.

## Credit task 2 - Zadání projektu

Projekt je souèástí vašeho závìreèného hodnocení. Nikdo by nemìl ztrácet èas a øešit svùj projekt a na konci semestru. Z tohoto dùvodu je tento "kreditní úkol" zamìøen na návrh vašeho projektu. Mùete pouít pøedchozí popis svého projektu jako základ, s tím rozdílem, e budou pøidány nezbytné detaily pro pøesnou implementaci.

Úkoly: 1. Popište kompletní datovou strukturu vašeho projektu. To znamená, e ji znáte téma projektu a musíte pøemıšlet o jeho potøebách (jakı typ pamìti, jak budou data uloena, jak budou vlákna pøistupovat k datùm atd.). 2. Popište algoritmus, kterı bude implementován, a jakı bude vısledek. Popište, jak bude algoritmus rozdìlen na kernely, metody na zaøízení nebo hostiteli a jak se tyto metody budou navzájem volat. Musíte zmínit všechny zdroje, které chcete pouít. 3. Pøipravte zprávu. Dokument musí bıt napsán tak podrobnì, aby jej mohl pouít jinı èlovìk a pøesnì vìdìl, o èem projekt je a jak bude zpracován.

## Credit task 3 - Simulace disease spread

- Postupujte podle pokynù z posledního semináøe.
- Poté se pokuste vyøešit tyto úkoly:
- Mapa svìta pouívá pouze nìkolik barev. Èerná pøedstavuje Slovensko a jeho hranice s okolními zemìmi. Zelená pøedstavuje naši zemi. Bílá pøedstavuje ostatní okolní zemì.
- Èervenı bod poblí mìsta Košice oznaèuje ohnisko slintavky a kulhavky.
- V kadém kroku simulace se èervená oblast rozšiøuje: kadı èernı pixel, kterı má alespoò jednoho èerveného souseda, se také zmìní na èervenı.
- Ulote obrázek, jakmile èervenı pixel dosáhne hranice s Èeskou republikou.

## Credit task 4 - Find pattern in image 

Najdìte pozici pattern0 a pattern1 v pùvodním obrázku (zdrojovém obrázku) a vrate pozici jeho levého horního rohu. Vzory mohou bıt otoèené!

# Projekt 

Projekt si mùe kadı zvolit podle sebe. Jde dìlat ve dvou. S kolegou jsme se domluvili na diferenciální evoluci. 

Mìl nìkolik vıtek:

- int r1_i, r2_i, r3_i mìly bıt unsigned int
- evaluate() by mìlo bıt constexpr, aby se pøi kompilaci vìdìlo, do které vìtve má vkroèit, takhle to vdy musí vyhodnocovat v runtime
- v kernelIndividual() poèitáme hranice uvnitø, to se mìlo poèítat venku pøi volání kernelu, takhle to je zbyteènı overhead, moná ještì nìco jiného
- prı by ještì nìjak upravil jednotlivé funkce pro poèitání fiteness, prı by tam dal ještì více pointrù, bitovıch operací a tak, aby to bylo rychlejší, co fakt nevymyslím ani za 100 let

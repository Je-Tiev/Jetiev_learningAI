# Chương 3: Biểu diễn tri thức và Lập luận (Knowledge Representation and Reasoning)

---

## Mục tiêu học tập

Sau khi hoàn thành chương này, người học sẽ có khả năng:
1. **Nắm vững nguyên lý Tác tử dựa trên tri thức:** Hiểu rõ kiến trúc Cơ sở tri thức (Knowledge Base - KB), cơ chế `TELL`/`ASK` và định nghĩa toán học của Hệ quả logic (Entailment: $KB \models \alpha$).
2. **Làm chủ Logic mệnh đề và Ngữ nghĩa hình thức:** Phân tích cú pháp, bảng chân trị, khái niệm mô hình (Model), tính khả quy (Satisfiability) và tính hằng đúng (Validity).
3. **Thành thạo các giải thuật suy diễn logic:** Cài đặt và phân biệt Suy diễn tiến (Forward Chaining), Suy diễn lùi (Backward Chaining) trên mệnh đề Horn; hiểu sâu thuật toán Hợp giải (Resolution) trên dạng chuẩn tắc hội (CNF).
4. **Mở rộng sang Logic vị từ bậc nhất (First-Order Logic - FOL):** Nắm vững vai trò của các lượng từ ($\forall, \exists$), kỹ thuật Skolem hóa (Skolemization) và thuật toán Đối khớp (Unification) để tự động hóa suy luận.
5. **Hiểu sâu cấu trúc Hệ chuyên gia và Giới hạn của AI ký hiệu:** Phân tích kiến trúc hệ chuyên gia (MYCIN, R1), cùng 3 tử huyệt kinh điển: Bùng nổ tổ hợp, Nút thắt thu thập tri thức và Độ giòn (Brittleness).
6. **Kết nối Logic với Đồ thị tri thức hiện đại:** Nắm bắt bước chuyển dịch từ logic hình thức sang Bản thể học (Ontology), Đồ thị tri thức (Knowledge Graph - $(h, r, t)$ triplets), nhúng đồ thị (TransE) và ứng dụng trong GraphRAG cho LLMs.

---

## 3.1 Tác tử dựa trên tri thức và Logic mệnh đề

### 3.1.1 Cơ sở tri thức (Knowledge Base - KB)
Một **tác tử dựa trên tri thức (Knowledge-Based Agent)** duy trì một cơ sở tri thức nội bộ chứa các câu (sentences) biểu diễn trạng thái của thế giới. Tác tử tương tác với KB thông qua hai hàm giao tiếp hình thức:
- **`TELL(KB, sentence)`:** Nạp thêm một khẳng định hoặc dữ kiện mới vào KB.
- **`ASK(KB, query)`:** Truy vấn xem một câu hỏi/khẳng định $\alpha$ có được suy ra từ KB hay không.

```
                   Nhận thức (Percepts)
                            │
                            ▼
                    ┌───────────────┐
                    │     TELL      │
                    └───────┬───────┘
                            ▼
               ┌────────────────────────┐
               │ Cơ sở tri thức (KB)    │
               │ • Luật nghiệp vụ       │
               │ • Sự kiện quan sát     │
               └────────────┬───────────┘
                            ▲
                    ┌───────┴───────┐
                    │      ASK      │ ◄─── Truy vấn (Query)
                    └───────┬───────┘
                            │
                            ▼
                    Hành động (Action)
```

#### Khái niệm Hệ quả logic (Entailment):
Một câu $\alpha$ là **hệ quả logic** của cơ sở tri thức $KB$ (ký hiệu: $KB \models \alpha$) khi và chỉ khi trong mọi mô hình (thế giới khả dĩ) mà $KB$ đúng, thì $\alpha$ cũng bắt buộc phải đúng:
$$KB \models \alpha \iff \mathcal{M}(KB) \subseteq \mathcal{M}(\alpha)$$
Trong đó $\mathcal{M}(KB)$ là tập hợp tất cả các mô hình thỏa mãn $KB$.

### 3.1.2 Cú pháp và Ngữ nghĩa hình thức của Logic mệnh đề
- **Cú pháp (Syntax):** Xây dựng từ các ký hiệu mệnh đề ($P, Q, R, \dots$) và 5 liên từ logic cơ bản:
  - Phủ định (Negation): $\neg P$ ("không phải $P$")
  - Hội (Conjunction): $P \land Q$ ("$P$ và $Q$")
  - Tuyển (Disjunction): $P \lor Q$ ("$P$ hoặc $Q$")
  - Kéo theo (Implication): $P \implies Q$ ("nếu $P$ thì $Q$"; tương đương $\neg P \lor Q$)
  - Tương đương (Biconditional): $P \iff Q$ ("$P$ khi và chỉ khi $Q$")

- **Ngữ nghĩa (Semantics) và Bảng chân trị:**
  Một **mô hình (Model)** $m$ là một phép gán giá trị chân lý $\{\text{True}, \text{False}\}$ cho từng ký hiệu mệnh đề. Với $n$ biến mệnh đề độc lập, tồn tại chính xác $2^n$ mô hình khả dĩ.

| $P$ | $Q$ | $\neg P$ | $P \land Q$ | $P \lor Q$ | $P \implies Q$ | $P \iff Q$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| False | False | True | False | False | **True** | True |
| False | True | True | False | True | **True** | False |
| True | False | False | False | True | **False** | False |
| True | True | False | True | True | **True** | True |

- **Tính hằng đúng (Validity / Tautology):** Một câu là hằng đúng nếu nó đúng trong *mọi* mô hình khả dĩ (ví dụ: $P \lor \neg P$).
- **Tính thỏa được (Satisfiability - SAT):** Một câu là thỏa được nếu tồn tại *ít nhất một* mô hình làm cho câu đó đúng. Bài toán SAT trong logic mệnh đề là bài toán NP-đầy đủ đầu tiên được chứng minh trong khoa học máy tính (Định lý Cook-Levin, 1971).

---

## 3.2 Lập luận và các Thuật toán Suy diễn (Inference Algorithms)

Một thuật toán suy diễn (ký hiệu $KB \vdash_i \alpha$: thuật toán $i$ suy ra $\alpha$ từ $KB$) cần thỏa mãn hai phẩm chất toán học:
1. **Tính vững chắc (Soundness / Truth-preserving):** Mọi điều suy ra đều phải đúng với thực tế:
   $$KB \vdash_i \alpha \implies KB \models \alpha$$
2. **Tính đầy đủ (Completeness):** Mọi chân lý là hệ quả logic đều phải được tìm thấy bởi thuật toán:
   $$KB \models \alpha \implies KB \vdash_i \alpha$$

### 3.2.1 Quy tắc suy diễn Modus Ponens
Quy tắc suy diễn nền tảng nhất: Nếu tiền đề $\alpha \implies \beta$ đúng và sự kiện $\alpha$ đúng, ta suy ra $\beta$ đúng:
$$\frac{\alpha \implies \beta, \quad \alpha}{\beta}$$

### 3.2.2 Mệnh đề Horn và Thuật toán Suy diễn tiến (Forward Chaining)
Một **mệnh đề Horn (Horn Clause)** là một tuyển của các literal trong đó có *nhiều nhất một literal khẳng định*:
$$\neg P_1 \lor \neg P_2 \lor \dots \lor \neg P_k \lor Q \quad \equiv \quad (P_1 \land P_2 \land \dots \land P_k) \implies Q$$
- **Cơ chế suy diễn tiến (Data-Driven):**
  1. Bắt đầu từ các sự kiện đã biết trong cơ sở tri thức.
  2. Tìm kiếm các luật có toàn bộ tiền đề đã được thỏa mãn.
  3. Kích hoạt luật và bổ sung hệ quả vào tập sự kiện đã biết.
  4. Lặp lại cho đến khi suy ra được câu hỏi mục tiêu hoặc không còn luật nào kích hoạt được.
- **Độ phức tạp:** Chạy trong thời gian **tuyến tính** $\mathcal{O}(\text{size}(KB))$, cực kỳ hiệu quả trong thực tế.

### 3.2.3 Thuật toán Suy diễn lùi (Backward Chaining)
- **Cơ chế (Goal-Driven):**
  1. Bắt đầu từ câu hỏi mục tiêu $Q$ cần chứng minh.
  2. Tìm trong KB các luật có hệ quả là $Q$.
  3. Kiểm tra xem các tiền đề của luật đó đã có trong sự kiện chưa; nếu chưa, đệ quy chứng minh từng tiền đề như các mục tiêu con (Sub-goals).
- **Ứng dụng:** Trọng tâm trong ngôn ngữ lập trình logic **Prolog** và các hệ thống chẩn đoán y tế, nơi ta muốn truy tìm nguyên nhân gây ra một triệu chứng cụ thể.

### 3.2.4 Dạng chuẩn tắc hội (CNF) và Thuật toán Hợp giải (Resolution)
Mọi câu trong logic mệnh đề đều có thể biến đổi tương đương về **Dạng chuẩn tắc hội (Conjunctive Normal Form - CNF)** — một phép hội của các mệnh đề tuyển (clause):
$$\mathcal{C}_1 \land \mathcal{C}_2 \land \dots \land \mathcal{C}_m \quad \text{với } \mathcal{C}_i = (l_{i,1} \lor l_{i,2} \lor \dots \lor l_{i,k})$$

#### Quy tắc Hợp giải (Resolution Rule):
Hai mệnh đề chứa hai literal đối ngẫu nhau ($P$ và $\neg P$) có thể được hợp giải thành một mệnh đề mới loại bỏ cả hai:
$$\frac{l_1 \lor \dots \lor l_i \lor P, \quad m_1 \lor \dots \lor m_j \lor \neg P}{l_1 \lor \dots \lor l_i \lor m_1 \lor \dots \lor m_j}$$

#### Chứng minh bằng phản chứng (Proof by Refutation):
Để chứng minh $KB \models \alpha$:
1. Giả sử điều ngược lại: Thêm câu phủ định $\neg \alpha$ vào cơ sở tri thức: $KB' = KB \land \neg \alpha$.
2. Chuyển đổi toàn bộ $KB'$ sang dạng CNF.
3. Liên tục áp dụng quy tắc Hợp giải lên các cặp mệnh đề.
4. Nếu sinh ra **mệnh đề rỗng (Empty clause / Falsehood $\square$)**, điều đó chứng minh $KB \land \neg \alpha$ mâu thuẫn $\implies$ Khẳng định $\alpha$ là đúng.
5. Thuật toán Hợp giải là **vững chắc và đầy đủ** cho logic mệnh đề.

---

## 3.3 Logic vị từ bậc nhất (First-Order Logic - FOL)

Logic mệnh đề bị giới hạn nghiêm trọng: Ta không thể diễn đạt một quy luật phổ quát cho một lớp đối tượng mà không phải liệt kê từng đối tượng cụ thể. Logic bậc nhất bổ sung năng lực biểu diễn phong phú hơn bằng cách mô hình hóa thế giới gồm:
- **Đối tượng (Objects / Constants):** Ví dụ: `John`, `XeHoi_1`, `So_5`.
- **Hàm (Functions):** Ánh xạ từ đối tượng sang đối tượng: `FatherOf(John)`, `Plus(2, 3)`.
- **Vị từ (Predicates / Relations):** Mối quan hệ giữa các đối tượng trả về True/False: `Brother(John, Bob)`, `Greater(5, 3)`.

### 3.3.1 Lượng từ trong FOL
1. **Lượng từ phổ dụng ($\forall$ - Universal Quantifier):** "Với mọi đối tượng $x$":
   $$\forall x \, (\text{Human}(x) \implies \text{Mortal}(x))$$
   *(Mọi người đều phải chết)*
2. **Lượng từ tồn tại ($\exists$ - Existential Quantifier):** "Tồn tại ít nhất một đối tượng $x$":
   $$\exists x \, (\text{Crown}(x) \land \text{OnHead}(x, \text{John}))$$
   *(Có một chiếc vương miện trên đầu John)*
3. **Tính đối ngẫu De Morgan giữa hai lượng từ:**
   $$\forall x \, \neg P(x) \equiv \neg \exists x \, P(x), \qquad \neg \forall x \, P(x) \equiv \exists x \, \neg P(x)$$

### 3.3.2 Skolem hóa (Skolemization) và Thuật toán Đối khớp (Unification)
Để máy tính tự động hóa suy luận trên FOL, ta cần chuyển FOL về dạng mệnh đề không lượng từ:
- **Skolem hóa:** Loại bỏ lượng từ tồn tại $\exists$:
  - Nếu $\exists x$ đứng độc lập: Thay bằng một **hằng số Skolem** mới: $\exists x \, \text{Rich}(x) \implies \text{Rich}(C_1)$.
  - Nếu $\exists y$ nằm trong phạm vi của $\forall x$: Đối tượng $y$ phụ thuộc vào $x$, ta thay bằng một **hàm Skolem** $f(x)$:
    $$\forall x \, \exists y \, \text{MotherOf}(y, x) \implies \forall x \, \text{MotherOf}(f(x), x)$$
- **Thuật toán Đối khớp (Unification - $\text{UNIFY}(p, q)$):**
  Tìm phép thế các biến (substitution) $\theta = \{v_1/t_1, v_2/t_2, \dots\}$ sao cho hai biểu thức vị từ trở nên đồng nhất:
  $$\text{SUBST}(\theta, p) = \text{SUBST}(\theta, q)$$
  Ví dụ:
  $$\text{UNIFY}(\text{Knows}(\text{John}, x), \text{Knows}(y, \text{Bill})) = \{y / \text{John}, x / \text{Bill}\}$$

---

## 3.4 Hệ thống dựa trên luật và Hệ chuyên gia (Expert Systems)

Thập niên 1980 chứng kiến sự nở rộ của các **Hệ chuyên gia (Expert Systems)** — đỉnh cao ứng dụng thương mại của AI ký hiệu.

```
┌─────────────────────────────────────────────────────────────────┐
│                      HỆ CHUYÊN GIA (EXPERT SYSTEM)               │
│                                                                 │
│   ┌────────────────────────┐         ┌──────────────────────┐   │
│   │ Bộ nhớ làm việc        │         │ Cơ sở luật           │   │
│   │ (Working Memory)       │         │ (Production Rules)   │   │
│   │ • Các sự kiện hiện tại │         │ • IF cond THEN act   │   │
│   └───────────┬────────────┘         └──────────┬───────────┘   │
│               │                                 │               │
│               └────────────────┬────────────────┘               │
│                                ▼                                │
│                     ┌────────────────────┐                      │
│                     │ Động cơ suy diễn   │                      │
│                     │ (Inference Engine) │                      │
│                     │ • Khớp (Match)     │                      │
│                     │ • Giải quyết xung  │                      │
│                     │   đột (Conflict)   │                      │
│                     │ • Thực thi (Act)   │                      │
│                     └─────────┬──────────┘                      │
│                               │                                 │
└───────────────────────────────┼─────────────────────────────────┘
                                ▼
                       Giải thích (Explanation)
```

### Các hệ chuyên gia kinh điển:
1. **MYCIN (1976 - Stanford):** Chẩn đoán các bệnh nhiễm trùng máu và viêm màng não. Chứa khoảng 600 luật và sử dụng **Hệ số tin cậy (Certainty Factors - CF $\in [-1, 1]$)** để xử lý tri thức không chắc chắn.
2. **DENDRAL (1965):** Phân tích quang phổ khối lượng để xác định cấu trúc phân tử hữu cơ.
3. **R1 / XCON (1980 - DEC):** Cấu hình tự động hệ thống máy tính VAX, giúp tiết kiệm cho tập đoàn DEC hơn 40 triệu USD mỗi năm.

---

## 3.5 Giới hạn của AI Ký hiệu

Dù đạt được nhiều thành tựu rực rỡ, AI ký hiệu vấp phải 3 "tử huyệt" nội tại khiến nó thoái trào trước làn sóng học máy:

### 3.5.1 Bùng nổ tổ hợp (Combinatorial Explosion)
Khi số lượng mệnh đề và thực thể tăng lên, không gian trạng thái cần tìm kiếm tăng theo hàm mũ $\mathcal{O}(2^n)$ hoặc $\mathcal{O}(b^d)$. Việc giải quyết các bài toán logic phức tạp vượt quá giới hạn năng lực tính toán của bất kỳ siêu máy tính nào.

### 3.5.2 Nút thắt thu thập tri thức (Knowledge Acquisition Bottleneck)
Phần lớn tri thức của chuyên gia con người là **tri thức ngầm (tacit knowledge)** được tích lũy qua hàng chục năm kinh nghiệm dưới dạng trực giác vô thức. Rất khó để một chuyên gia phát biểu toàn bộ trực giác của mình thành hàng ngàn luật `IF-THEN` tường minh mà không bỏ sót ngoại lệ.

### 3.5.3 Độ giòn (Brittleness)
Các hệ thống logic thuần túy không có khả năng "suy biến mềm" (graceful degradation):
- Nếu gặp một tình huống hơi khác một chút so với các luật đã định nghĩa, hệ thống lập tức bế tắc (trả về kết quả rỗng).
- **Nguyên lý nổ tung (Principle of Explosion):** Trong logic cổ điển, chỉ cần xuất hiện một mâu thuẫn nhỏ ($P \land \neg P$), toàn bộ hệ thống có thể chứng minh được mọi điều vô lý ($P \land \neg P \vdash Q, \forall Q$).

---

## 3.6 Từ Logic đến Đồ thị Tri thức (From Logic to Knowledge Graphs)

Để khắc phục độ giòn của logic mệnh đề nhưng vẫn bảo tồn tính minh bạch và cấu trúc ngữ nghĩa, AI hiện đại đã phát triển sang **Bản thể học (Ontology)** và **Đồ thị tri thức (Knowledge Graph - KG)**.

```
                  ┌───────────────────────────────┐
                  │    ĐỒ THỊ TRI THỨC (KG)       │
                  │                               │
                  │   [Albert Einstein]           │
                  │           │                   │
                  │           │ sinh_ra_tai       │
                  │           ▼                   │
                  │         [Ulm] ──nam_tai──► [Đức]
                  │           ▲                   │
                  │           │ que_huong         │
                  │   [Robert Mayer]              │
                  └───────────────────────────────┘
```

### 3.6.1 Bản thể học (Ontology)
Bản thể học định nghĩa một hệ thống phân cấp chuẩn mực về các khái niệm (Classes), các mối quan hệ (Properties) và các ràng buộc trong một lĩnh vực cụ thể. Sử dụng các chuẩn Web ngữ nghĩa như **RDF (Resource Description Framework)** và **OWL (Web Ontology Language)**.

### 3.6.2 Đồ thị Tri thức và Biểu diễn Bộ ba (Triplets)
- Một đồ thị tri thức $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ lưu trữ tri thức dưới dạng các **bộ ba (Triplets)**:
  $$(h, r, t) \in \mathcal{E}_{\text{entities}} \times \mathcal{R}_{\text{relations}} \times \mathcal{E}_{\text{entities}}$$
  Trong đó $h$ là thực thể đầu (Head), $r$ là quan hệ (Relation), $t$ là thực thể đuôi (Tail).
  *Ví dụ:* `(Barack_Obama, BornIn, Honolulu)`.

### 3.6.3 Nhúng Đồ thị tri thức (Knowledge Graph Embeddings - KGE)
Để kết hợp đồ thị tri thức với mạng nơ-ron sâu, các thực thể và quan hệ được ánh xạ vào không gian vector liên tục $\mathbb{R}^d$:
- **Mô hình TransE (Bordes et al., 2013):** Coi quan hệ $r$ như một phép tịnh tiến từ $h$ đến $t$:
  $$\mathbf{h} + \mathbf{r} \approx \mathbf{t}$$
  Hàm mất mát khoảng cách: $d(\mathbf{h} + \mathbf{r}, \mathbf{t}) = \|\mathbf{h} + \mathbf{r} - \mathbf{t}\|_{1/2}$.
- **Mô hình ComplEx / RotatE:** Coi quan hệ như phép quay trong không gian số phức để biểu diễn các quan hệ đối xứng, phản xạ và nghịch đảo.

### 3.6.4 Đồ thị tri thức trong Kỷ nguyên LLMs (GraphRAG)
Mặc dù LLM sở hữu lượng tri thức khổng lồ nhưng hay gặp hiện tượng "ảo giác" (Hallucination) và thiếu tri thức chuyên sâu cập nhật. **GraphRAG (Graph Retrieval-Augmented Generation)** kết hợp sức mạnh biểu diễn ngôn ngữ của LLM với đồ thị tri thức:
1. Truy vấn của người dùng được phân tích để tìm các thực thể liên quan trên Knowledge Graph.
2. Trích xuất cây quan hệ nhiều bước (Multi-hop Reasoning Subgraph).
3. Đưa đồ thị có cấu trúc vào làm ngữ cảnh chính xác cho LLM suy luận và trả lời.

---

## 3.7 Bảng tổng kết đối chiếu: Logic hình thức vs. Đồ thị tri thức

| Tiêu chí | Logic hình thức (FOL / Horn) | Đồ thị tri thức (Knowledge Graph) |
| :--- | :--- | :--- |
| **Đơn vị biểu diễn** | Câu logic, vị từ $P(x, y)$, mệnh đề CNF | Nút thực thể (Nodes) và Cạnh quan hệ (Edges) $(h, r, t)$ |
| **Cơ chế suy diễn** | Hợp giải (Resolution), Modus Ponens | Lan truyền trên đồ thị, Link Prediction, GNN, RAG |
| **Tính linh hoạt** | Kém, nhạy cảm với lỗi cú pháp và mâu thuẫn | Rất cao, dễ dàng mở rộng và bổ sung dữ liệu mới |
| **Tích hợp học máy** | Rất khó nhúng vào không gian vector | Rất tự nhiên qua các thuật toán nhúng (TransE, Node2Vec) |
| **Khả năng giải thích** | Dẫn xuất logic 100% minh bạch từng bước | Truy vết đường dẫn trên đồ thị tri thức (Path-based reasoning) |

---

## 3.8 Tài liệu tham khảo (References)

1. **Russell, S., & Norvig, P. (2020).** *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. (Chương 7: Logical Agents; Chương 8: First-Order Logic; Chương 9: Inference in First-Order Logic; Chương 12: Knowledge Representation).
2. **Mitchell, T. M. (1997).** *Machine Learning*. McGraw-Hill. [Tham khảo: `resources/Machine Learning - Tom Mitchell.pdf`] (Chương 10: Learning Sets of Rules).
3. **Robinson, J. A. (1965).** A machine-oriented logic based on the resolution principle. *Journal of the ACM*, 12(1), 23-41. (Công trình gốc khai sinh thuật toán Hợp giải).
4. **Shortliffe, E. H. (1976).** *Computer-Based Medical Consultations: MYCIN*. Elsevier. (Tài liệu gốc về hệ chuyên gia MYCIN).
5. **Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013).** Translating embeddings for modeling multi-relational data. *Advances in Neural Information Processing Systems (NeurIPS)*, 26. (Bài báo nền tảng về mô hình nhúng đồ thị TransE).
6. **Edge, D., et al. (2024).** From Local to Global: A Graph RAG Approach to Query-Focused Summarization. *arXiv preprint arXiv:2404.16130*.

---

## 3.9 Định hướng thực hành (`demo.ipynb`)

Trong tệp [`demo.ipynb`](./demo.ipynb), chúng ta xây dựng hệ thống thực nghiệm kết nối giữa **Suy luận logic cổ điển** và **Đồ thị tri thức hiện đại**:

1. **Thành phần 1 - Động cơ suy diễn tiến (Forward Chaining Engine):**
   - Định nghĩa lớp `HornClause`: $\text{Premises} \implies \text{Conclusion}$.
   - Cài đặt thuật toán suy diễn tiến trên cơ sở tri thức chẩn đoán lỗi máy tính (IT Support Troubleshooting KB).
2. **Thành phần 2 - Động cơ suy diễn lùi (Backward Chaining Engine):**
   - Truy vết mục tiêu con theo cơ chế Goal-driven để trả lời câu hỏi: *"Máy in có bị hỏng phần cứng hay không?"*.
3. **Thành phần 3 - Chứng minh định lý bằng thuật toán Hợp giải (Resolution Refutation):**
   - Chuyển đổi các mệnh đề về dạng chuẩn tắc CNF.
   - Triển khai hàm hợp giải hai mệnh đề và kiểm tra mệnh đề rỗng $\square$ để chứng minh một kết luận logic.
4. **Thành phần 4 - Xây dựng Đồ thị tri thức (Knowledge Graph Engine):**
   - Biểu diễn đồ thị tri thức dưới dạng các bộ ba $(h, r, t)$ trên miền phả hệ và sự nghiệp các nhà khoa học máy tính lỗi lạc (Turing, McCarthy, Hinton, LeCun, Bengio).
   - Cài đặt hàm suy luận quan hệ nhiều bước (Multi-hop Reasoning): Suy luận quan hệ bắc cầu `IsColleagueOf` hoặc `InfluencedBy`.
5. **Thành phần 5 - Trực quan hóa Đồ thị tri thức:**
   - Sử dụng `networkx` và `matplotlib` để trực quan hóa đồ thị tri thức với các nút thực thể và nhãn quan hệ trên cạnh.

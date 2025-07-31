# 決定木のスクラッチ実装
import numpy as np
from collections import Counter

class MyDecisionTreeClassifier:
    # 決定木の分類器
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Parameters:
        max_depth: 木の最大深さ
        min_samples_split: 分割に必要な最小サンプル数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    class Node:
        # 決定木のノードを表すクラス
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            """
            feature: 分割に使用する特徴量のインデックス
            threshold: 分割に使用する閾値
            left: 左の子ノード
            right: 右の子ノード
            value: ノードのクラスラベルの予測値（葉ノードの場合）
            """
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
    
    def fit(self, X, y):
        """
        訓練データを用いて決定木を構築する
        X: 特徴量の行列（numpy配列）
        y: クラスラベルのベクトル（numpy配列）
        """
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        """
        決定木を成長させるための再帰関数
        Parameters:
        X: 特徴量の行列（numpy配列）
        y: クラスラベルのベクトル（numpy配列）
        depth: 現在の深さ

        Returns:
        node: 決定木のノード
        """
        n_samples, n_features = X.shape
        n_classes = len(set(y))

        # 停止条件をチェック
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split or n_classes == 1:
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)
        
        # 最適な分割を見つける
        best_feature, best_threshold = self._best_split(X, y)

        # データを分割
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs # bit演算子のnot

        # 子ノードを再帰的に構築
        left_subtree = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_subtree = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return self.Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)
    
    def _best_split(self, X, y):
        """
        最適な分割を見つける関数
        Parameters:
        X: 特徴量の行列（numpy配列）
        y: クラスラベルのベクトル（numpy配列）

        Returns:
        best_feature: 最適な特徴量のインデックス
        best_threshold: 最適な閾値
        """
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature in range(self.n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                # 情報利得を計算
                # IG(T, a) = I(T) - Sum_{v in V} (|T_v| / |T|) * I(T_v)
                gain = self._information_gain(X[:, feature], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, X_column, y, threshold):
        """
        情報利得を計算する関数
        Parameters:
        X_column: 特徴量の列（numpy配列）
        y: クラスラベルのベクトル（numpy配列）
        threshold: 分割に使用する閾値

        Returns:
        gain: 情報利得
        """
        parent_entropy = self._entropy(y)

        left_idxs = X_column < threshold
        right_idxs = ~left_idxs

        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0
        
        n = len(y)
        n_left, n_right = len(y[left_idxs]), len(y[right_idxs])
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])

        # 加重平均を計算
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        gain = parent_entropy - child_entropy
        return gain
    
    def _entropy(self, y):
        """
        エントロピーを計算する関数
        Parameters:
        y: クラスラベルのベクトル（numpy配列）

        Returns:
        entropy: エントロピー
        """
        hist = Counter(y)
        probabilities = np.array(list(hist.values())) / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
        return entropy
    
    def _most_common_label(self, y):
        """
        最も頻出のクラスラベルを返す関数
        Parameters:
        y: クラスラベルのベクトル（numpy配列）

        Returns:
        most_common: 最も頻出のクラスラベル
        """
        return Counter(y).most_common(1)[0][0]
    
    def predict(self, X):
        """
        新しいデータに対して予測を行う関数
        Parameters:
        X: 特徴量の行列（numpy配列）

        Returns:
        predictions: 予測されたクラスラベルのベクトル（numpy配列）
        """
        predictions = np.array([self._traverse_tree(x, self.tree) for x in X])
        return predictions
    
    def _traverse_tree(self, x, node):
        """
        決定木をトラバースして予測を行う関数
        Parameters:
        x: 特徴量ベクトル (numpy配列)
        node: 現在のノード
        
        Returns:
        value: 予測されたクラスラベル
        """
        if node.value is not None:
            return node.value
        
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
        return node.value

if __name__ == "__main__":
    # テスト用のコード
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    iris = load_iris()    
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = MyDecisionTreeClassifier(max_depth=3, min_samples_split=2)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accucacy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accucacy:.4f}") # 0.9667
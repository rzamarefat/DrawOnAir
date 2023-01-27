import cv2
import mediapipe as mp

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands

class PatternRecognizer:
    def __init__(self):
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._mp_hands = mp.solutions.hands
        self._selected_nodes = []
        self._selected_nodes_id = []
        
        self._last_index_finger_coord = None

        self._video_width, self._video_height = 640, 480
        self._gt_path = [f for f in reversed([7, 3, 2, 1])] #start index from 0
    
    def _make_gt_visible(self, image, nodes):
        for index, nod_num in enumerate(self._gt_path):
            try:
                cv2.line(image, nodes[nod_num][0], nodes[self._gt_path[index+1]][0], (0, 225, 0), 2)    
            except:
                continue
    
    def _read_stream(self, id_):
        cap = cv2.VideoCapture(id_)

        return cap

    def _calc_nodes(self, frame_w, frame_h):
        node_0 =  ((int(frame_h * 0.44), int(frame_w * 0.1)), 0)
        node_1 =  ((int(frame_h * 0.66), int(frame_w * 0.1)), 1)
        node_2 =  ((int(frame_h * 0.88), int(frame_w * 0.1)), 2)
        node_3 =  ((int(frame_h * 0.44), int(frame_w * 0.3)), 3)
        node_4 =  ((int(frame_h * 0.66), int(frame_w * 0.3)), 4)
        node_5 =  ((int(frame_h * 0.88), int(frame_w * 0.3)), 5)
        node_6 =  ((int(frame_h * 0.44), int(frame_w * 0.5)), 6)
        node_7 =  ((int(frame_h * 0.66), int(frame_w * 0.5)), 7)
        node_8 =  ((int(frame_h * 0.88), int(frame_w * 0.5)), 8)
        
        nodes = [node_0, node_1, node_2, node_3, node_4, node_5, node_6, node_7, node_8, node_8]

        return nodes
            
    def _draw_nodes(self, image, nodes):
        for node in nodes:
            cv2.circle(image, node[0], 20, (233, 129, 55), -1)


    def _draw_landmarks(self, image, hand_landmarks):
        
        self._mp_drawing.draw_landmarks(
            image,
            hand_landmarks,

            self._mp_hands.HAND_CONNECTIONS,
            self._mp_drawing_styles.get_default_hand_landmarks_style(),
            self._mp_drawing_styles.get_default_hand_connections_style()
            )

    def _set_path(self, nodes, x_index_finger, y_index_finger):
        
        node_0, node_1, node_2, node_3, node_4, node_5, node_6, node_7, node_8 = nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], nodes[6], nodes[7], nodes[8]

        if (abs(node_0[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_0[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_0[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_0)
                self._selected_nodes_id.append(0)

        if (abs(node_1[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_1[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_1[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_1)
                self._selected_nodes_id.append(1)

        if (abs(node_2[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_2[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_2[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_2)
                self._selected_nodes_id.append(2)

        if (abs(node_3[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_3[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_3[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_3)
                self._selected_nodes_id.append(3)

        if (abs(node_4[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_4[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_4[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_4)
                self._selected_nodes_id.append(4)
                
        if (abs(node_5[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_5[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_5[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_5)
                self._selected_nodes_id.append(5)


        if (abs(node_6[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_6[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_6[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_6)
                self._selected_nodes_id.append(6)

        if (abs(node_7[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_7[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_7[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_7)
                self._selected_nodes_id.append(7)
                
        if (abs(node_8[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_8[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_8[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_8)
                self._selected_nodes_id.append(8)


            
    def handle_stream(self):
        cap = self._read_stream(-1)
        
        nodes = self._calc_nodes(self._video_width, self._video_height)

        with self._mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                is_authorized = False
                jojo = list(set([f[1] for f in reversed(self._selected_nodes)]))
                if  jojo == self._gt_path:
                    is_authorized = True
                    

                success, image = cap.read()
                
                
                if not(is_authorized):
                    self._make_gt_visible(image, nodes)

                
                if is_authorized:
                    cv2.putText(image, "Authorized", (200, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (120, 110, 225), 2)

                
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if not success:
                    continue

                results = hands.process(image)
                
                image_width = image.shape[1]
                image_height = image.shape[0]

                if not(is_authorized):
                    self._draw_nodes(image, nodes)
                

                
                    for node, _ in self._selected_nodes:
                        cv2.circle(image, node, 20, (10, 100, 200), -1)
                

                
                    if len(self._selected_nodes) >= 2:
                        for index, (s, _) in enumerate(self._selected_nodes):
                            try:
                                cv2.line(image, s, self._selected_nodes[index+1][0], (200, 100, 150), 2)    
                            except:
                                continue

                    if len(self._selected_nodes) > 0:
                        cv2.line(image, self._selected_nodes[-1][0], self._last_index_finger_coord, (200, 100, 150), 2)


                
                image.flags.writeable = False
                

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # print(hand_landmarks)
                        self._draw_landmarks(image, hand_landmarks)

                        x_index_finger = int(hand_landmarks.landmark[self._mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
                        y_index_finger = int(hand_landmarks.landmark[self._mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)

                        self._last_index_finger_coord = (x_index_finger, y_index_finger)

                        if not(is_authorized):
                            self._set_path(nodes, x_index_finger, y_index_finger)
                        

                cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cap.release()


if __name__ == "__main__":
    pr = PatternRecognizer()
    pr.handle_stream()

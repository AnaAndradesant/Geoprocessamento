import json, os
from qgis.PyQt.QtWidgets import (
    QAction, QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem,
    QFrame, QApplication, QMessageBox
)
from qgis.PyQt.QtCore import Qt, QObject, QEvent
from qgis.PyQt.QtGui import QFont, QColor, QKeySequence, QIcon
from qgis.core import (
    QgsProject, QgsMessageLog, Qgis,
    QgsLayerTreeGroup, QgsLayerTreeLayer
)

PLUGIN_DIR = os.path.dirname(__file__)


class ArrowKeyFilter(QObject):
    def __init__(self, get_widget_fn, toggle_fn):
        super().__init__()
        self._get_widget = get_widget_fn
        self._toggle = toggle_fn

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Down:
                self._get_widget().navigate(1)
                return True
            elif event.key() == Qt.Key_Up:
                self._get_widget().navigate(-1)
                return True
            elif event.key() == Qt.Key_F9:
                self._toggle()
                return True
        return False


class ConnectPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.dock = None
        self.action = None
        self.action_toggle = None
        self._key_filter = None
        self._status_label = None

    def initGui(self):
        icon = QIcon(os.path.join(PLUGIN_DIR, 'icon.png'))
        self.action = QAction(icon, 'Connect', self.iface.mainWindow())
        self.action.setCheckable(True)
        self.action.setToolTip('Abrir/Fechar Connect  (F9)')
        self.action.triggered.connect(self.toggle_dock)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu('&Connect', self.action)

        self.action_toggle = QAction('Toggle Connect', self.iface.mainWindow())
        self.action_toggle.setShortcut(QKeySequence('F9'))
        self.action_toggle.triggered.connect(self._toggle_by_shortcut)
        self.iface.mainWindow().addAction(self.action_toggle)

        self._create_dock()

        self._key_filter = ArrowKeyFilter(lambda: self.dock.widget(), self._toggle_by_shortcut)
        QApplication.instance().installEventFilter(self._key_filter)

        self._status_label = QLabel('  Connect: inativo  ')
        self._status_label.setStyleSheet('color:#7f8c8d;font-size:10px;padding:2px 6px;')
        self.iface.mainWindow().statusBar().insertWidget(0, self._status_label)

        self.iface.layerTreeView().contextMenuAboutToShow.connect(self._add_context_menu)
        QgsProject.instance().readProject.connect(self._on_project_read)
        QgsProject.instance().writeProject.connect(self._on_project_write)

    def unload(self):
        if self._key_filter:
            QApplication.instance().removeEventFilter(self._key_filter)
            self._key_filter = None
        if self._status_label:
            self.iface.mainWindow().statusBar().removeWidget(self._status_label)
            self._status_label = None
        try:
            self.iface.layerTreeView().contextMenuAboutToShow.disconnect(self._add_context_menu)
            QgsProject.instance().readProject.disconnect(self._on_project_read)
            QgsProject.instance().writeProject.disconnect(self._on_project_write)
        except Exception:
            pass
        self.iface.mainWindow().removeAction(self.action_toggle)
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu('&Connect', self.action)
        if self.dock:
            self.iface.removeDockWidget(self.dock)
            self.dock.deleteLater()
            self.dock = None

    def toggle_dock(self, checked):
        if self.dock:
            self.dock.setVisible(checked)

    def _toggle_by_shortcut(self):
        if self.dock:
            vis = self.dock.isVisible()
            self.dock.setVisible(not vis)
            self.action.setChecked(not vis)

    def _create_dock(self):
        self.dock = QDockWidget('Connect', self.iface.mainWindow())
        self.dock.setObjectName('ConnectDock')
        self.dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        widget = ConnectWidget(self.iface, self._update_status)
        self.dock.setWidget(widget)
        self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.setVisible(False)
        self.dock.visibilityChanged.connect(self.action.setChecked)

    def _add_context_menu(self, menu):
        w = self.dock.widget()
        view = self.iface.layerTreeView()
        selected_nodes = view.selectedNodes()
        if not selected_nodes:
            return
        existing_names = set(g['name'] for g in w.nav_groups)
        groups_to_add = [n for n in selected_nodes if isinstance(n, QgsLayerTreeGroup) and n.name() not in existing_names]
        groups_to_remove = [n for n in selected_nodes if isinstance(n, QgsLayerTreeGroup) and n.name() in existing_names]
        menu.addSeparator()
        if groups_to_add:
            lbl = 'Connect: Adicionar grupo' if len(groups_to_add) == 1 else 'Connect: Adicionar %d grupos' % len(groups_to_add)
            act = QAction(lbl, menu)
            act.triggered.connect(lambda: [w.add_group(g) for g in groups_to_add])
            menu.addAction(act)
        if groups_to_remove:
            lbl = 'Connect: Remover grupo' if len(groups_to_remove) == 1 else 'Connect: Remover %d grupos' % len(groups_to_remove)
            act2 = QAction(lbl, menu)
            act2.triggered.connect(lambda: [w.remove_group(g.name()) for g in groups_to_remove])
            menu.addAction(act2)

    def _update_status(self, text, active):
        if self._status_label:
            self._status_label.setText(text)
            if active:
                self._status_label.setStyleSheet(
                    'color:white;background:#27ae60;font-size:10px;'
                    'padding:2px 10px;border-radius:3px;font-weight:bold;'
                )
            else:
                self._status_label.setStyleSheet('color:#7f8c8d;font-size:10px;padding:2px 6px;')

    def _on_project_read(self):
        self.dock.widget().load_from_project()

    def _on_project_write(self):
        self.dock.widget().save_to_project()


class ConnectWidget(QWidget):
    def __init__(self, iface, status_cb):
        super().__init__()
        self.iface = iface
        self._status_cb = status_cb
        self.nav_groups = []
        self.current_index = -1
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(8, 8, 8, 8)

        header = QFrame()
        header.setStyleSheet('background:#2c3e50;border-radius:6px;')
        hl = QVBoxLayout(header)
        hl.setContentsMargins(8, 8, 8, 8)
        title = QLabel('Connect')
        title.setFont(QFont('Arial', 13, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('color:white;letter-spacing:2px;')
        hl.addWidget(title)
        h1 = QLabel('Clique direito no grupo > Adicionar ao Connect')
        h1.setAlignment(Qt.AlignCenter)
        h1.setStyleSheet('color:#f39c12;font-size:9px;')
        hl.addWidget(h1)
        h2 = QLabel('Setas cima/baixo navegam  |  F9 abre/fecha')
        h2.setAlignment(Qt.AlignCenter)
        h2.setStyleSheet('color:#2ecc71;font-size:9px;')
        hl.addWidget(h2)
        main_layout.addWidget(header)

        lbl = QLabel('Grupos no Connect  (arraste para reordenar):')
        lbl.setStyleSheet('color:#7f8c8d;font-size:9px;margin-top:2px;')
        main_layout.addWidget(lbl)

        self.group_list = QListWidget()
        self.group_list.setDragDropMode(QListWidget.InternalMove)
        self.group_list.setDefaultDropAction(Qt.MoveAction)
        self.group_list.setMinimumHeight(200)
        self.group_list.setStyleSheet(
            'QListWidget{border:1px solid #bdc3c7;border-radius:4px;}'
            'QListWidget::item{padding:8px;border-bottom:1px solid #ecf0f1;font-size:11px;}'
            'QListWidget::item:selected{background:#d6eaf8;color:#2c3e50;}'
        )
        self.group_list.itemDoubleClicked.connect(self._jump_to_group)
        main_layout.addWidget(self.group_list)

        btn_row = QHBoxLayout()
        self.btn_remove = QPushButton('Remover')
        self.btn_remove.setStyleSheet(
            'QPushButton{border:1px solid #e74c3c;color:#e74c3c;border-radius:3px;padding:5px;}'
            'QPushButton:hover{background:#fadbd8;}'
        )
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear = QPushButton('Limpar tudo')
        self.btn_clear.setStyleSheet(
            'QPushButton{border:1px solid #bdc3c7;border-radius:3px;padding:5px;}'
            'QPushButton:hover{background:#ecf0f1;}'
        )
        self.btn_clear.clicked.connect(self._clear_all)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_clear)
        main_layout.addLayout(btn_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet('color:#bdc3c7;')
        main_layout.addWidget(sep)

        self.lbl_current = QLabel('Nenhum grupo ativo')
        self.lbl_current.setAlignment(Qt.AlignCenter)
        self.lbl_current.setWordWrap(True)
        self.lbl_current.setMinimumHeight(40)
        self.lbl_current.setStyleSheet(
            'background:#95a5a6;color:white;border-radius:5px;padding:8px;font-weight:bold;font-size:11px;'
        )
        main_layout.addWidget(self.lbl_current)

        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton('Anterior')
        self.btn_prev.setMinimumHeight(40)
        self.btn_prev.setStyleSheet(
            'QPushButton{background:#3498db;color:white;border-radius:5px;font-weight:bold;font-size:11px;}'
            'QPushButton:hover{background:#2980b9;}'
        )
        self.btn_prev.clicked.connect(lambda: self.navigate(-1))
        self.btn_next = QPushButton('Proxima')
        self.btn_next.setMinimumHeight(40)
        self.btn_next.setStyleSheet(
            'QPushButton{background:#e67e22;color:white;border-radius:5px;font-weight:bold;font-size:11px;}'
            'QPushButton:hover{background:#d35400;}'
        )
        self.btn_next.clicked.connect(lambda: self.navigate(1))
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        main_layout.addLayout(nav_layout)
        main_layout.addStretch()

    def add_group(self, group_node):
        name = group_node.name()
        if any(g['name'] == name for g in self.nav_groups):
            return
        n_layers = len(group_node.findLayers())
        self.nav_groups.append({'name': name})
        # Oculta o grupo ao adicionar
        self._set_group_visible(name, False)
        self._sync_ui()
        self.iface.messageBar().pushMessage(
            'Connect',
            "Grupo '%s' adicionado (%d camadas)." % (name, n_layers),
            level=Qgis.Success, duration=2
        )

    def remove_group(self, name):
        self.nav_groups = [g for g in self.nav_groups if g['name'] != name]
        if self.current_index >= len(self.nav_groups):
            self.current_index = len(self.nav_groups) - 1
        self._sync_ui()
        if not self.nav_groups:
            self.current_index = -1
            self._reset_label()

    def _remove_selected(self):
        row = self.group_list.currentRow()
        if 0 <= row < len(self.nav_groups):
            name = self.nav_groups[row]['name']
            if row == self.current_index:
                self._set_group_visible(name, False)
            self.remove_group(name)

    def _clear_all(self):
        for g in self.nav_groups:
            self._set_group_visible(g['name'], False)
        self.nav_groups = []
        self.current_index = -1
        self._sync_ui()
        self._reset_label()

    def _set_group_visible(self, group_name, visible):
        root = QgsProject.instance().layerTreeRoot()
        group_node = root.findGroup(group_name)
        if group_node:
            group_node.setItemVisibilityChecked(visible)

    def _sync_ui(self):
        ui_names = [self.group_list.item(i).data(Qt.UserRole) for i in range(self.group_list.count())]
        if ui_names:
            ntg = {g['name']: g for g in self.nav_groups}
            reordered = [ntg[n] for n in ui_names if n in ntg]
            for g in self.nav_groups:
                if g not in reordered:
                    reordered.append(g)
            self.nav_groups = reordered
        self.group_list.clear()
        for i, g in enumerate(self.nav_groups):
            item = QListWidgetItem()
            if i == self.current_index:
                item.setText('  [ATIVO]  %s' % g['name'])
                item.setBackground(QColor('#d5f5e3'))
                item.setForeground(QColor('#1e8449'))
            else:
                item.setText('  %s' % g['name'])
                item.setForeground(QColor('#7f8c8d'))
            item.setData(Qt.UserRole, g['name'])
            self.group_list.addItem(item)

    def navigate(self, direction):
        self._sync_ui()
        if not self.nav_groups:
            self.iface.messageBar().pushMessage(
                'Connect', 'Adicione grupos com botao direito primeiro.',
                level=Qgis.Warning, duration=3
            )
            return
        if 0 <= self.current_index < len(self.nav_groups):
            self._set_group_visible(self.nav_groups[self.current_index]['name'], False)
        if self.current_index < 0:
            self.current_index = 0 if direction == 1 else len(self.nav_groups) - 1
        else:
            self.current_index = (self.current_index + direction) % len(self.nav_groups)
        self._show_current()

    def _show_current(self):
        if self.current_index < 0 or self.current_index >= len(self.nav_groups):
            return
        current = self.nav_groups[self.current_index]
        self._set_group_visible(current['name'], True)
        self.iface.mapCanvas().refresh()
        self.iface.mapCanvas().redrawAllLayers()
        name = current['name']
        pos = '%d/%d' % (self.current_index + 1, len(self.nav_groups))
        self.lbl_current.setText('[%s]  %s' % (pos, name))
        self.lbl_current.setStyleSheet(
            'background:#2ecc71;color:white;border-radius:5px;padding:8px;font-weight:bold;font-size:11px;'
        )
        self._status_cb('  %s  ' % name, True)
        self._sync_ui()
        self.group_list.setCurrentRow(self.current_index)

    def _jump_to_group(self, item):
        row = self.group_list.currentRow()
        if 0 <= row < len(self.nav_groups):
            if 0 <= self.current_index < len(self.nav_groups):
                self._set_group_visible(self.nav_groups[self.current_index]['name'], False)
            self.current_index = row
            self._show_current()

    def _reset_label(self):
        self.lbl_current.setText('Nenhum grupo ativo')
        self.lbl_current.setStyleSheet(
            'background:#95a5a6;color:white;border-radius:5px;padding:8px;font-weight:bold;font-size:11px;'
        )
        self._status_cb('  Connect: inativo  ', False)

    def save_to_project(self):
        QgsProject.instance().writeEntry('Connect', 'nav_groups', json.dumps(self.nav_groups))
        QgsProject.instance().writeEntry('Connect', 'current_index', str(self.current_index))

    def load_from_project(self):
        data, ok = QgsProject.instance().readEntry('Connect', 'nav_groups', '')
        if ok and data:
            try:
                self.nav_groups = json.loads(data)
            except Exception:
                self.nav_groups = []
        idx, ok2 = QgsProject.instance().readEntry('Connect', 'current_index', '-1')
        self.current_index = int(idx) if ok2 else -1
        self._sync_ui()
        if 0 <= self.current_index < len(self.nav_groups):
            name = self.nav_groups[self.current_index]['name']
            pos = '%d/%d' % (self.current_index + 1, len(self.nav_groups))
            self.lbl_current.setText('[%s]  %s' % (pos, name))
            self.lbl_current.setStyleSheet(
                'background:#2ecc71;color:white;border-radius:5px;padding:8px;font-weight:bold;font-size:11px;'
            )
            self._status_cb('  %s  ' % name, True)
